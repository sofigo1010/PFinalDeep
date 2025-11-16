"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Header from "../../components/layout/header";
import Footer from "../../components/layout/footer";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { TrendingUp, BarChart3, Info } from "lucide-react";

export default function ResultsPage() {
  const router = useRouter();
  const [predictionData, setPredictionData] = useState([]);

  useEffect(() => {
    const raw = sessionStorage.getItem("predictions");
    if (!raw) {
      router.push("/upload");
      return;
    }
    try {
      const data = JSON.parse(raw);
      // esperamos un array de { fecha, lstm, ensemble }
      setPredictionData(
        data.map((p) => ({
          fecha: p.fecha,
          "Quantile LSTM": p.lstm,
          Ensemble: p.ensemble,
        }))
      );
    } catch {
      console.error("Error parseando predicciones desde sessionStorage");
      router.push("/upload");
    }
  }, [router]);

  return (
    <div className="min-h-screen bg-black text-white">
      <Header />
      <main className="py-20">
        <div className="max-w-7xl mx-auto px-4">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold mb-4">Resultados de Predicción</h1>
            <p className="text-xl text-gray-400">
              Análisis completo de tus predicciones de ventas
            </p>
          </div>

          {/* Gráfica de líneas */}
          <div className="bg-gray-900 rounded-xl p-8 border border-gray-800 mb-8">
            <div className="flex items-center gap-2 mb-6">
              <TrendingUp className="h-6 w-6 text-blue-500" />
              <h2 className="text-2xl font-bold">Gráfica de Predicciones</h2>
            </div>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={predictionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="fecha"
                    stroke="#9CA3AF"
                    tick={{ fill: "#9CA3AF" }}
                  />
                  <YAxis stroke="#9CA3AF" tick={{ fill: "#9CA3AF" }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1F2937",
                      border: "1px solid #374151",
                      borderRadius: "8px",
                      color: "#FFFFFF",
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="Quantile LSTM"
                    stroke="#3B82F6"
                    strokeWidth={2}
                    name="Quantile LSTM"
                  />
                  <Line
                    type="monotone"
                    dataKey="Ensemble"
                    stroke="#F59E0B"
                    strokeWidth={2}
                    name="Ensemble"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Explicación de métricas */}
          <div className="bg-gray-900 rounded-xl p-8 border border-gray-800 mb-8">
            <div className="flex items-center gap-2 mb-6">
              <Info className="h-6 w-6 text-blue-500" />
              <h2 className="text-2xl font-bold">¿Qué significan estas métricas?</h2>
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-bold text-blue-400 mb-2">
                    Quantile LSTM
                  </h3>
                  <p className="text-gray-300">
                    Predicciones generadas por el modelo LSTM usando cuantiles.
                  </p>
                </div>
              </div>
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-bold text-yellow-400 mb-2">
                    Ensemble
                  </h3>
                  <p className="text-gray-300">
                    Combinación de Prophet y XGB para mayor robustez.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Tabla de predicciones */}
          <div className="bg-gray-900 rounded-xl p-8 border border-gray-800 mb-8">
            <div className="flex items-center gap-2 mb-6">
              <BarChart3 className="h-6 w-6 text-blue-500" />
              <h2 className="text-2xl font-bold">Tabla de Predicciones</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="pb-3 text-gray-300 font-bold">Fecha</th>
                    <th className="pb-3 text-blue-400 font-bold">LSTM</th>
                    <th className="pb-3 text-yellow-400 font-bold">Ensemble</th>
                  </tr>
                </thead>
                <tbody>
                  {predictionData.map((row, idx) => (
                    <tr key={idx} className="border-b border-gray-800">
                      <td className="py-3 text-white font-mono">{row.fecha}</td>
                      <td className="py-3 text-blue-300 font-mono">
                        {row["Quantile LSTM"].toFixed(2)}
                      </td>
                      <td className="py-3 text-yellow-300 font-mono">
                        {row.Ensemble.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Botón para métricas */}
          <div className="text-center">
            <button
              onClick={() => router.push("/metrics")}
              className="bg-purple-600 hover:bg-purple-700 text-white px-8 py-4 rounded-lg font-bold text-lg flex items-center gap-2 mx-auto"
            >
              <BarChart3 className="h-5 w-5" />
              Ver métricas comparativas
            </button>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
