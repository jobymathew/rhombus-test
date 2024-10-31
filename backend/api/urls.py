from django.urls import path
from .views import DataTypeInferenceView

urlpatterns = [
    path('infer-types/', DataTypeInferenceView.as_view(), name='infer-types'),
]