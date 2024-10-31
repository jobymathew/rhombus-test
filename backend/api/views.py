import tempfile
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from scripts.data_type_inference import DataTypeInference

class DataTypeInferenceView(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request):
        if 'file' not in request.FILES:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

        file_obj = request.FILES['file']
        
        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.name)[1]) as tmp_file:
            for chunk in file_obj.chunks():
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name

        try:
            # Initialize the DataTypeInference class
            inference = DataTypeInference(
                categorical_threshold=0.5,
                date_sample_size=1000,
                memory_efficient=True
            )

            # Process the file
            df = inference.process_file(tmp_file_path)

            # Get column types
            column_types = df.dtypes.apply(lambda x: str(x)).to_dict()
            
            # Map pandas types to user-friendly names
            type_mapping = {
                'object': 'Text',
                'int64': 'Integer',
                'Int64': 'Integer (with nulls)',
                'float64': 'Decimal',
                'datetime64[ns]': 'Date/Time',
                'bool': 'Boolean',
                'category': 'Category',
                'string': 'Text (optimized)'
            }

            friendly_types = {
                col: type_mapping.get(str(dtype), str(dtype))
                for col, dtype in column_types.items()
            }

            # Get a preview of the data
            preview_data = df.head(5).to_dict('records')

            return Response({
                'column_types': friendly_types,
                'preview_data': preview_data,
                'total_rows': len(df)
            })

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)