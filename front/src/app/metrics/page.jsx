"use client"

import Header from "../../components/layout/header"
import Footer from "../../components/layout/footer"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"
import { TrendingUp, CheckCircle, Info, Target } from "lucide-react"

// Datos de métricas
const metricsData = [
  { metric: "MAE", Prophet: 899.18, Quantile: 0.13, Ensemble: 245.78 },
  { metric: "RMSE", Prophet: 1198.29, Quantile: 0.24, Ensemble: 344.22 },
  { metric: "MAPE", Prophet: 126.4, Quantile: 17.6, Ensemble: 48.52 },
  { metric: "sMAPE", Prophet: 46.6, Quantile: 13.48, Ensemble: 20.43 },
  { metric: "R²", Prophet: 0.15, Quantile: 0.342, Ensemble: 0.93 },
]

// Tarjeta de métrica
const MetricCard = ({ title, value, isGood = true }) => (
  <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 text-center">
    <h3 className="text-lg font-bold text-gray-300 mb-2">{title}</h3>
    <div className="flex items-center justify-center gap-2">
      <span className="text-3xl font-bold text-white">{value}</span>
      {isGood && <CheckCircle className="h-6 w-6 text-green-500" />}
    </div>
  </div>
)

export default function MetricsPage() {
  // Filtrar solo las métricas que queremos en cards
  const topMetrics = metricsData.filter(({ metric }) =>
    ["MAE", "RMSE", "MAPE", "R²"].includes(metric)
  )

  return (
    <div className="min-h-screen bg-black text-white">
      <Header />
      <main className="py-20">
        <div className="max-w-7xl mx-auto px-4">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-white mb-4">
              Métricas de Evaluación
            </h1>
            <p className="text-xl text-gray-400">
              Análisis comparativo del desempeño de los modelos
            </p>
          </div>

          {/* Tarjetas de métricas destacadas (Quantile LSTM) */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-12">
            {topMetrics.map(({ metric, Quantile }) => {
              const displayValue =
                metric === "MAPE"
                  ? `${Quantile.toFixed(2)}%`
                  : metric === "R²"
                  ? Quantile.toFixed(3)
                  : Quantile
              return (
                <MetricCard
                  key={metric}
                  title={metric}
                  value={displayValue}
                />
              )
            })}
          </div>

          {/* Tabla comparativa */}
          <div className="bg-gray-900 rounded-xl p-8 border border-gray-800 mb-8">
            <div className="flex items-center gap-2 mb-6">
              <Target className="h-6 w-6 text-blue-500" />
              <h2 className="text-2xl font-bold text-white">
                Comparación de Modelos
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
                      <td className="py-3 text-red-300 font-mono">
                        {["MAPE", "sMAPE"].includes(metric)
                          ? `${Prophet.toFixed(2)}%`
                          : Prophet}
                      </td>
                      <td className="py-3 text-yellow-300 font-mono">
                        {["MAPE", "sMAPE"].includes(metric)
                          ? `${Quantile.toFixed(2)}%`
                          : Quantile}
                      </td>
                      <td className="py-3 text-green-300 font-mono">
                        {["MAPE", "sMAPE"].includes(metric)
                          ? `${Ensemble.toFixed(2)}%`
                          : Ensemble}
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
                    valores reales. Menor es mejor.
                  </p>
                </div>

                <div>
                  <h3 className="text-lg font-bold text-purple-400 mb-2">
                    RMSE (Raíz del Error Cuadrático Medio)
                  </h3>
                  <p className="text-gray-300">
                    Penaliza más los errores grandes. Útil para detectar
                    predicciones muy alejadas de la realidad.
                  </p>
                </div>

                <div>
                  <h3 className="text-lg font-bold text-yellow-400 mb-2">
                    MAPE (Error Porcentual Absoluto Medio)
                  </h3>
                  <p className="text-gray-300">
                    Error expresado como porcentaje. Fácil de interpretar.
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
                    valores reales son muy pequeños.
                  </p>
                </div>

                <div>
                  <h3 className="text-lg font-bold text-red-400 mb-2">
                    R² (Coeficiente de Determinación)
                  </h3>
                  <p className="text-gray-300">
                    Mide qué tan bien el modelo explica la variabilidad de los
                    datos. 1.0 es perfecto, 0.0 es malo.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Gráfico de barras */}
          <div className="bg-gray-900 rounded-xl p-8 border border-gray-800 mb-8">
            <div className="flex items-center gap-2 mb-6">
              <TrendingUp className="h-6 w-6 text-blue-500" />
              <h2 className="text-2xl font-bold text-white">
                Comparación Visual
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
                  <YAxis
                    stroke="#9CA3AF"
                    tick={{ fill: "#9CA3AF" }}
                  />
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

          {/* Interpretación */}
          <div className="bg-gradient-to-r from-green-900/20 to-blue-900/20 rounded-xl p-8 border border-green-500/30">
            <div className="text-center">
              <div className="text-4xl mb-4">🎯</div>
              <h2 className="text-2xl font-bold text-white mb-4">
                Interpretación de Resultados
              </h2>
              <p className="text-xl text-green-300 font-bold mb-2">
                El Ensemble reduce el MAE en ≈ 73 % respecto a Prophet y un 49 % respecto a Quantile LSTM.
              </p>
              <p className="text-gray-300">
                Con un R² de 0.93, el modelo Ensemble explica el 93 % de la variabilidad en tus datos de ventas.
              </p>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  )
}
