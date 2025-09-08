from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Sample data store
items = {}

class Item(BaseModel):
    name: str
    price: float

@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]

@app.post("/items/{item_id}")
def create_item(item_id: int, item: Item):
    if item_id in items:
        raise HTTPException(status_code=400, detail="Item already exists")
    items[item_id] = item
    return item

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    items[item_id] = item
    return item

@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    del items[item_id]
    return {"detail": "Item deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)