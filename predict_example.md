# Example: Test the /predict endpoint

You can test the `/predict/` endpoint using `curl` to upload an image file. Replace `example.jpg` with the path to your test image:

```powershell
curl -X POST "http://localhost:8000/predict/" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@example.jpg;type=image/jpeg"
```

You can also use the interactive Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs) to upload an image and see the prediction result.
