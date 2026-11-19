import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:5000"
});

export const predictImage = (formData) =>
  API.post("/predict-image", formData);

export const predictLifestyle = (data) =>
  API.post("/predict-lifestyle", data);

export const getAdvice = (data) =>
  API.post("/get-advice", data);