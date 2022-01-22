# sketch2estampe - demo

## Usage

```
docker build -t sketch2estampe .
docker run -d -p 5000:5000 -v "$(pwd)"/outputs:/app/outputs --name sketch2estampe sketch2estampe
```

```
docker run sketch2estampe
docker stop sketch2estampe
```