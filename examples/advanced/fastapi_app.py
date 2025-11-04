import modal

# Create an image with FastAPI installed
image = modal.Image.debian_slim().pip_install("fastapi[standard]")

app = modal.App("example-fastapi", image=image)


# Create a FastAPI application
@app.function()
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    
    web_app = FastAPI()
    
    @web_app.get("/")
    def root():
        return {"message": "Hello from Modal FastAPI!"}
    
    @web_app.get("/items/{item_id}")
    def read_item(item_id: int, q: str = None):
        return {"item_id": item_id, "q": q}
    
    @web_app.post("/calculate")
    def calculate(data: dict):
        operation = data.get("operation", "add")
        a = data.get("a", 0)
        b = data.get("b", 0)
        
        if operation == "add":
            result = a + b
        elif operation == "multiply":
            result = a * b
        elif operation == "power":
            result = a ** b
        else:
            result = None
        
        return {"operation": operation, "a": a, "b": b, "result": result}
    
    return web_app


# To deploy this as a web service:
# modal deploy fastapi_app.py
# This will give you a public URL that you can access


