"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import Papa from "papaparse";
import Header from "../../components/layout/header";
import Footer from "../../components/layout/footer";
import { Upload, FileText, ChevronDown, Loader2 } from "lucide-react";

export default function UploadPage() {
  const [selectedMonths, setSelectedMonths] = useState(6);
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const router = useRouter();

  const monthOptions = [1, 6, 12, 24];
  const requiredCols = ["fecha", "ventas_previas", "otras_vars"];

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped && (dropped.type === "text/csv" || dropped.name.endsWith(".csv"))) {
      setFile(dropped);
    }
  };

  const handleFileChange = (e) => {
    const picked = e.target.files?.[0];
    if (picked && (picked.type === "text/csv" || picked.name.endsWith(".csv"))) {
      setFile(picked);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;
    setIsLoading(true);

    try {
      // 1) Parsear CSV
      const text = await file.text();
      const { data, errors, meta } = Papa.parse(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
      });
      if (errors.length) {
        console.error("Parse errors:", errors);
        throw new Error("Error al parsear el CSV");
      }

      let historical = [];

      // 2) Si es CSV de entrenamiento original: agrupar por "Order Date" y sumar "Sales"
      if (meta.fields.includes("Order Date") && meta.fields.includes("Sales")) {
        const agg = data.reduce((acc, row) => {
          const date = new Date(
            row["Order Date"].replace(/(\d+)\/(\d+)\/(\d+)/, "$3-$2-$1")
          )
            .toISOString()
            .slice(0, 10);
          const sale = parseFloat(row.Sales) || 0;
          acc[date] = (acc[date] || 0) + sale;
          return acc;
        }, {});
        historical = Object.entries(agg).map(([fecha, ventas_previas]) => ({
          fecha,
          ventas_previas,
          otras_vars: 0,
        }));
      } else {
        // 3) Validar que tenga exactamente las columnas requeridas
        const missing = requiredCols.filter((c) => !meta.fields.includes(c));
        if (missing.length) {
          throw new Error(
            `Faltan columnas: ${missing.join(
              ", "
            )}. El CSV debe contener: ${requiredCols.join(", ")}.`
          );
        }
        // 4) Mapear directamente
        historical = data.map((row) => ({
          fecha: row.fecha,
          ventas_previas: row.ventas_previas,
          otras_vars: row.otras_vars,
        }));
      }

      const horizon = selectedMonths * 30; // aprox. 30 días/mes

      // 5) Llamada a la API
      const res = await fetch("http://127.0.0.1:8000/predict-horizon", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ historical, horizon }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Error en la respuesta de la API");
      }
      const { predictions } = await res.json();

      // 6) Guardar en sessionStorage y navegar a resultados
      sessionStorage.setItem("predictions", JSON.stringify(predictions));
      router.push("/results");
    } catch (err) {
      console.error(err);
      alert("Falló la predicción: " + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white">
      <Header />
      <main className="py-20">
        <div className="max-w-4xl mx-auto px-4">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold mb-4">Configurar Predicción</h1>
            <p className="text-xl text-gray-400">
              Configura los parámetros y sube tus datos de ventas
            </p>
          </div>

          {/* Selector de meses */}
          <div className="mb-8">
            <label className="block text-lg font-bold mb-4">
              ¿Cuántos meses deseas predecir?
            </label>
            <div className="relative">
              <select
                value={selectedMonths}
                onChange={(e) => setSelectedMonths(Number(e.target.value))}
                className="w-full bg-gray-900 border border-gray-600 px-4 py-3 rounded-lg appearance-none focus:border-blue-500"
              >
                {monthOptions.map((m) => (
                  <option key={m} value={m}>
                    {m} {m === 1 ? "mes" : "meses"}
                  </option>
                ))}
              </select>
              <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400 pointer-events-none" />
            </div>
          </div>

          {/* Área de carga de archivo */}
          <div className="mb-8">
            <div
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              className={`bg-gray-900 border-2 border-dashed rounded-xl p-12 text-center space-y-6 transition-colors ${
                dragActive
                  ? "border-blue-500 bg-blue-500/10"
                  : file
                  ? "border-green-500 bg-green-500/10"
                  : "border-gray-600 hover:border-blue-500"
              }`}
            >
              <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-blue-600">
                <Upload className="h-10 w-10 text-white" />
              </div>
              <h3 className="text-xl font-bold">
                {file ? file.name : "Arrastra tu archivo CSV aquí"}
              </h3>
              <p className="text-gray-400">
                o haz clic para seleccionar desde tu computadora
              </p>
              <input
                id="file-upload"
                type="file"
                accept=".csv"
                className="hidden"
                onChange={handleFileChange}
              />
              <label
                htmlFor="file-upload"
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-bold inline-flex items-center gap-2 cursor-pointer"
              >
                <FileText className="h-5 w-5" />
                Cargar CSV
              </label>
              <div className="text-gray-500">Formatos soportados: CSV</div>
            </div>
          </div>

          {/* Instrucciones */}
          <div className="mb-8 bg-gray-900 border border-gray-700 rounded-lg p-6">
            <h3 className="text-lg font-bold mb-3">Formato requerido del CSV:</h3>
            <p className="text-gray-300 mb-4">
              Columnas:{" "}
              <span className="text-blue-400 font-mono">
                fecha, ventas_previas, otras_vars
              </span>
            </p>
            <pre className="bg-black p-4 rounded-lg font-mono text-sm text-gray-200">
{`fecha,ventas_previas,otras_vars
2023-01-01,5000,variable1
2023-01-02,5200,variable2`}
            </pre>
          </div>

          {/* Preview del archivo */}
          {file && (
            <div className="mb-8 bg-gray-900 border border-gray-700 rounded-lg p-6">
              <h3 className="text-lg font-bold mb-3">Archivo seleccionado:</h3>
              <div className="flex items-center gap-3 text-gray-300">
                <FileText className="h-5 w-5 text-blue-400" />
                <span>{file.name}</span>
                <span>({(file.size / 1024).toFixed(1)} KB)</span>
              </div>
            </div>
          )}

          {/* Botón de envío */}
          <div className="text-center">
            <button
              onClick={handleSubmit}
              disabled={!file || isLoading}
              className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-8 py-4 rounded-lg text-lg font-bold flex items-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Procesando predicción...
                </>
              ) : (
                "Enviar a predicción"
              )}
            </button>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
