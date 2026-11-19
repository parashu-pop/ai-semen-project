import { BrowserRouter, Routes, Route } from "react-router-dom";
import ImageAnalysis from "./pages/ImageAnalysis";
import LifestyleAnalysis from "./pages/LifestyleAnalysis";
import Home from "./pages/Home";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/image" element={<ImageAnalysis />} />
        <Route path="/lifestyle" element={<LifestyleAnalysis />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;