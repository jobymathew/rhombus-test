# Rhombus Test Project

  

This project is a web application designed to process and display data with a primary focus on data type inference and conversion for datasets. Built with Django for the backend, Pandas for data processing, and React for the frontend, this app automatically infers data types, allowing users to upload, process, and visualize CSV/Excel files.

  

## Key Script: `data_type_inference.py`

Located in `backend/scripts`, `data_type_inference.py` is the core data processing engine of this application. It:

- Uses Pandas to intelligently infer and convert data types, ensuring that mixed data types, dates, and large datasets are accurately handled.

- Solves common issues like defaulting to `object` dtype, enabling a high degree of accuracy in type conversion.

  

## Prerequisites

-  **Docker** (optional if running without Docker)

-  **Python 3.8+**

-  **Node.js** (for frontend if running without Docker)

  

## How to Run with Docker

  

1.  **Clone the Repository:**

```bash
git clone https://github.com/jobymathew/rhombus-test.git

cd rhombus-test
```
2. **Build and Run the Docker Containers:**
```bash
docker-compose up --build
```
3. **Access the Application**

The application should now be running at http://localhost:8000.

## How to Run Without Docker

1. **Backend Setup (Django)**


 - Navigate to the backend folder:

```bash
cd backend
```

 - Install the required Python packages:


```bash
pip install -r requirements.txt
```

 - Run database migrations:


```bash
pyton manage.py migrate
```

 - Start the Django server:

```bash
python manage.py runserver
```

2. **Frontend Setup (React)**


 - Open a new terminal and navigate to the frontend folder:

```bash
cd frontend
```

 - Install the necessary packages:
```bash
npm install
```

 - Start the React development server:
```bash
npm start
```

3. **Access the Application**
- The frontend should be accessible at http://localhost:3000.

## Additional Notes

 - **Data Processing**: `data_type_inference.py` is critical for data accuracy and optimized performance, especially for large datasets.
