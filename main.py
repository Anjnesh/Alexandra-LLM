from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from process_a import process_a1, process_a2
from process_b import process_b
from process_c import process_c
import uvicorn

app = FastAPI()

@app.post("/analyze")
async def analyze_image(image_file: UploadFile = File(...), heatmap_file: UploadFile = File(...)):
    image_bytes = await image_file.read()
    heatmap_bytes = await heatmap_file.read()

    # Process A1
    description_a1 = process_a1(image_bytes)

    # Process A2
    salient_elements_a2 = process_a2(image_bytes, heatmap_bytes)

    # Process B
    cognitive_load_b = process_b(image_bytes)

    # Process C
    final_output = process_c(description_a1, salient_elements_a2, cognitive_load_b)

    return JSONResponse(content=final_output)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
