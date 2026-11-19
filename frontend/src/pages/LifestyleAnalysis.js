import React, { useState } from "react";
import axios from "axios";
import "./PageStyles.css";

function LifestyleAnalysis() {
  const [form, setForm] = useState({
    age: "",
    height: "",
    weight: "",
    sleep: "",
    smoking: "No",
    alcohol: "No",
    stress: "Low",
  });

  const [result, setResult] = useState("");

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    try {
      const res = await axios.post(
        "http://127.0.0.1:5000/predict-lifestyle",
        form
      );
      setResult(res.data.result);
    } catch (error) {
      console.error(error);
      alert("Backend error");
    }
  };

  return (
    <div className="page-container">
      <div className="card">
        <h1>🏃 Lifestyle Analysis</h1>

        {/* QUESTIONS UI */}

        <label>Age (years)</label>
        <input
          name="age"
          placeholder="Enter your age"
          onChange={handleChange}
        />

        <label>Height (cm)</label>
        <input
          name="height"
          placeholder="Enter your height"
          onChange={handleChange}
        />

        <label>Weight (kg)</label>
        <input
          name="weight"
          placeholder="Enter your weight"
          onChange={handleChange}
        />

        <label>How many hours do you sleep daily?</label>
        <input
          name="sleep"
          placeholder="e.g. 7-8 hours"
          onChange={handleChange}
        />

        <label>Do you smoke?</label>
        <select name="smoking" onChange={handleChange}>
          <option value="No">No</option>
          <option value="Yes">Yes</option>
        </select>

        <label>Do you consume alcohol?</label>
        <select name="alcohol" onChange={handleChange}>
          <option value="No">No</option>
          <option value="Yes">Yes</option>
        </select>

        <label>What is your stress level?</label>
        <select name="stress" onChange={handleChange}>
          <option value="Low">Low</option>
          <option value="Moderate">Moderate</option>
          <option value="High">High</option>
        </select>

        <button onClick={handleSubmit}>Analyze Lifestyle</button>

        {/* RESULT UI */}

        {result && (
          <div className="result">
            <h2>Fertility Risk Level: {result}</h2>

            {result === "Low" && (
              <p>✅ Your lifestyle is healthy. Keep it up!</p>
            )}
            {result === "Medium" && (
              <p>⚠️ Moderate risk. Improve sleep, diet, and habits.</p>
            )}
            {result === "High" && (
              <p>🚨 High risk. Consider lifestyle changes immediately.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default LifestyleAnalysis;