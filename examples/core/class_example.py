import modal

image = modal.Image.debian_slim().pip_install("numpy")

app = modal.App("example-class", image=image)


# Use a class to maintain state across function calls
@app.cls(cpu=2)
class ModelInference:
    """
    A class-based Modal function that can maintain state.
    Useful for loading models once and reusing them.
    """
    
    @modal.enter()
    def setup(self):
        """This runs once when the container starts"""
        import numpy as np
        print("Loading model...")
        # In a real scenario, you would load a ML model here
        self.model = np.random.rand(100, 100)
        print("Model loaded!")
    
    @modal.method()
    def predict(self, x: float):
        """This method can be called multiple times without reloading the model"""
        import numpy as np
        # Simulate inference
        result = float(np.mean(self.model) * x)
        print(f"Prediction for {x}: {result}")
        return result
    
    @modal.method()
    def batch_predict(self, inputs: list):
        """Process multiple inputs at once"""
        # Inside the class, we can call methods directly
        results = []
        for x in inputs:
            import numpy as np
            result = float(np.mean(self.model) * x)
            print(f"Prediction for {x}: {result}")
            results.append(result)
        return results
    
    @modal.exit()
    def cleanup(self):
        """This runs when the container shuts down"""
        print("Cleaning up...")


@app.local_entrypoint()
def main():
    # Create an instance of the class
    model = ModelInference()
    
    # Call methods on the instance
    result1 = model.predict.remote(5.0)
    print(f"Result 1: {result1}")
    
    result2 = model.predict.remote(10.0)
    print(f"Result 2: {result2}")
    
    # Batch prediction
    results = model.batch_predict.remote([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Batch results: {results}")


