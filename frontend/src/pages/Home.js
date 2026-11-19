import React from "react";
import { useNavigate } from "react-router-dom";
import "./Home.css";

function Home() {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      <h1 className="title">🧬 Fertility AI System</h1>

      <div className="card-container">
        <div className="card" onClick={() => navigate("/image")}>
          <h2>🧪 Image Analysis</h2>
          <p>Analyze sperm image using AI</p>
        </div>

        <div className="card" onClick={() => navigate("/lifestyle")}>
          <h2>🏃 Lifestyle Analysis</h2>
          <p>Check fertility based on lifestyle</p>
        </div>
      </div>
    </div>
  );
}

export default Home;