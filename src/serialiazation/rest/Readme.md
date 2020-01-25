
run server with command:
```
$  python app.py
```

classify the image dog.jpg with command:
```
$  curl -X POST -F image=@dog.jpg http://localhost:5000/predict
```

result should be in json format:
```
$  curl -X POST -F image=@dog.jpg http://localhost:5000/predict
```

