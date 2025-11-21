"use client";

import Header from "../../components/layout/header";
import Footer from "../../components/layout/footer";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { TrendingUp, CheckCircle, Info, Target } from "lucide-react";

// Métricas últimos 90 días (backtest in-sample)
const metricsData = [
  { metric: "MAE",   Prophet: 671.91, Quantile: 591.24, Ensemble: 72.62 },
  { metric: "RMSE",  Prophet: 953.65, Quantile: 1021.83, Ensemble: 98.84 },
  { metric: "MAPE",  Prophet: 120.31, Quantile: 70.28,  Ensemble: 15.12 },
  { metric: "sMAPE", Prophet: 62.50,  Quantile: 55.05,  Ensemble: 12.64 },
  { metric: "R²",    Prophet: 0.263,  Quantile: 0.154,  Ensemble: 0.992 },
];

// Tarjeta de métrica (usada para destacar el Ensemble)
const MetricCard = ({ title, value, isGood = true }) => (
  <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 text-center">
    <h3 className="text-lg font-bold text-gray-300 mb-2">{title}</h3>
    <div className="flex items-center justify-center gap-2">
      <span className="text-3xl font-bold text-white">{value}</span>
      {isGood && <CheckCircle className="h-6 w-6 text-green-500" />}
    </div>
  </div>
);

export default function MetricsPage() {
  // Usamos solo las métricas clave para las cards, tomando SIEMPRE la del Ensemble
  const topMetrics = metricsData.filter(({ metric }) =>
    ["MAE", "RMSE", "MAPE", "R²"].includes(metric)
  );

  return (
    <div className="min-h-screen bg-black text-white">
      <Header />
      <main className="py-20">
        <div className="max-w-7xl mx-auto px-4">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-white mb-4">
              Métricas de Evaluación (últimos 90 días)
            </h1>
            <p className="text-xl text-gray-400">
              Comparación del desempeño de Prophet, LSTM cuantil y Ensemble
            </p>
          </div>

          {/* Tarjetas de métricas destacadas (Ensemble) */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-12">
            {topMetrics.map(({ metric, Ensemble }) => {
              const displayValue =
                metric === "MAPE"
                  ? `${Ensemble.toFixed(2)}%`
                  : metric === "R²"
                  ? Ensemble.toFixed(3)
                  : Ensemble.toFixed(2);

              return (
                <MetricCard
                  key={metric}
                  title={`${metric} (Ensemble)`}
                  value={displayValue}
                />
              );
            })}
          </div>

          {/* Tabla comparativa */}
          <div className="bg-gray-900 rounded-xl p-8 border border-gray-800 mb-8">
            <div className="flex items-center gap-2 mb-6">
              <Target className="h-6 w-6 text-blue-500" />
              <h2 className="text-2xl font-bold text-white">
                Comparación de Modelos (últimos 90 días)
              </h2>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="pb-3 text-gray-300 font-bold">Métrica</th>
                    <th className="pb-3 text-red-400 font-bold">Prophet</th>
                    <th className="pb-3 text-yellow-400 font-bold">
                      Quantile LSTM
                    </th>
                    <th className="pb-3 text-green-400 font-bold">Ensemble</th>
                  </tr>
                </thead>
                <tbody>
                  {metricsData.map(({ metric, Prophet, Quantile, Ensemble }) => (
                    <tr
                      key={metric}
                      className="border-b border-gray-800 last:border-0"
                    >
                      <td className="py-3 text-white font-bold">{metric}</td>

                      {/* Prophet */}
                      <td className="py-3 text-red-300 font-mono">
                        {["MAPE", "sMAPE"].includes(metric)
                          ? `${Prophet.toFixed(2)}%`
                          : metric === "R²"
                          ? Prophet.toFixed(3)
                          : Prophet.toFixed(2)}
                      </td>

                      {/* Quantile LSTM */}
                      <td className="py-3 text-yellow-300 font-mono">
                        {["MAPE", "sMAPE"].includes(metric)
                          ? `${Quantile.toFixed(2)}%`
                          : metric === "R²"
                          ? Quantile.toFixed(3)
                          : Quantile.toFixed(2)}
                      </td>

                      {/* Ensemble */}
                      <td className="py-3 text-green-300 font-mono">
                        {["MAPE", "sMAPE"].includes(metric)
                          ? `${Ensemble.toFixed(2)}%`
                          : metric === "R²"
                          ? Ensemble.toFixed(3)
                          : Ensemble.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Explicación de métricas */}
          <div className="bg-gray-900 rounded-xl p-8 border border-gray-800 mb-8">
            <div className="flex items-center gap-2 mb-6">
              <Info className="h-6 w-6 text-blue-500" />
              <h2 className="text-2xl font-bold text-white">
                ¿Qué significan estas métricas?
              </h2>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-bold text-blue-400 mb-2">
                    MAE (Error Absoluto Medio)
                  </h3>
                  <p className="text-gray-300">
                    Promedio de las diferencias absolutas entre predicciones y
                    valores reales. Valores más bajos indican un modelo más
                    preciso.
                  </p>
                </div>

                <div>
                  <h3 className="text-lg font-bold text-purple-400 mb-2">
                    RMSE (Raíz del Error Cuadrático Medio)
                  </h3>
                  <p className="text-gray-300">
                    Penaliza más los errores grandes. Útil para detectar si el
                    modelo se equivoca mucho en picos de demanda.
                  </p>
                </div>

                <div>
                  <h3 className="text-lg font-bold text-yellow-400 mb-2">
                    MAPE (Error Porcentual Absoluto Medio)
                  </h3>
                  <p className="text-gray-300">
                    Error promedio expresado como porcentaje sobre las ventas
                    reales. Fácil de interpretar para negocio.
                  </p>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-bold text-green-400 mb-2">
                    sMAPE (Error Porcentual Absoluto Simétrico)
                  </h3>
                  <p className="text-gray-300">
                    Versión simétrica del MAPE que evita sesgos cuando los
                    valores reales son muy pequeños o muy grandes.
                  </p>
                </div>

                <div>
                  <h3 className="text-lg font-bold text-red-400 mb-2">
                    R² (Coeficiente de Determinación)
                  </h3>
                  <p className="text-gray-300">
                    Mide qué proporción de la variabilidad de las ventas queda
                    explicada por el modelo. 1.0 es perfecto; valores cercanos
                    a 0 indican poco poder explicativo.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Gráfico de barras comparativo */}
          <div className="bg-gray-900 rounded-xl p-8 border border-gray-800 mb-8">
            <div className="flex items-center gap-2 mb-6">
              <TrendingUp className="h-6 w-6 text-blue-500" />
              <h2 className="text-2xl font-bold text-white">
                Comparación visual de modelos
              </h2>
            </div>

            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="metric"
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
                  <Bar dataKey="Prophet" fill="#EF4444" name="Prophet" />
                  <Bar dataKey="Quantile" fill="#FBBF24" name="Quantile LSTM" />
                  <Bar dataKey="Ensemble" fill="#10B981" name="Ensemble" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Interpretación final */}
          <div className="bg-gradient-to-r from-green-900/20 to-blue-900/20 rounded-xl p-8 border border-green-500/30">
            <div className="text-center">
              <div className="text-4xl mb-4">🎯</div>
              <h2 className="text-2xl font-bold text-white mb-4">
                Interpretación de resultados
              </h2>
              <p className="text-xl text-green-300 font-bold mb-2">
                El Ensemble reduce el MAE en ~89&nbsp;% frente a Prophet y ~88&nbsp;% frente al LSTM cuantil,
                y también logra el menor MAPE (≈15&nbsp;%).
              </p>
              <p className="text-gray-300">
                Con un R² de {metricsData.find(m => m.metric === "R²")?.Ensemble.toFixed(3)}{" "}
                el Ensemble explica prácticamente la totalidad de la variabilidad
                de las ventas en esta ventana de 90 días, convirtiéndose en el
                modelo más fiable para tomar decisiones de inventario y
                planeación.
              </p>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
