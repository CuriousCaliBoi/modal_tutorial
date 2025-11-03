import sys
import modal

app = modal.App("example-hello-world")


@app.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)
    return i * i


@app.local_entrypoint()
def main():
    # Run the function locally
    print(f.local(1000))

    # Run the function remotely on Modal
    print(f.remote(1000))

    # Run the function in parallel and remotely on Modal
    total = 0
    for ret in f.map(range(200)):
        total += ret
    
    print(total)


