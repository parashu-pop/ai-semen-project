import React, { useState } from "react";
import axios from "axios";
import "./PageStyles.css";

function ImageAnalysis() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");

  const handleSubmit = async () => {
    if (!file) {
      alert("Please select an image");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(
        "http://127.0.0.1:5000/predict-image",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      // ✅ RESULT + CONFIDENCE
      setResult(`${res.data.result} (${res.data.confidence}%)`);

    } catch (error) {
      console.error(error);
      alert("Backend error");
    }
  };

  return (
    <div className="page-container">
      <div className="card">
        <h1>🧪 Image Analysis</h1>

        {/* File Input */}
        <input
          type="file"
          onChange={(e) => setFile(e.target.files[0])}
        />

        {/* Button */}
        <button onClick={handleSubmit}>
          Analyze Image
        </button>

        {/* Result Display */}
        {result && (
          <h2 className="result">
            Result: {result}
          </h2>
        )}
      </div>
    </div>
  );
}

export default ImageAnalysis;