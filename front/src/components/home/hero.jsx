import Link from "next/link"
import { ArrowRight, Zap, BarChart3, TrendingUp } from "lucide-react"

export default function Hero() {
  return (
    <section className="bg-black py-20">
      <div className="max-w-7xl mx-auto px-4">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-8">
            <h1 className="text-5xl lg:text-6xl font-bold text-white leading-tight">
              <span className="text-blue-500">SalesVision AI</span>
              <br />
              Predice tu futuro comercial
            </h1>

            <p className="text-xl text-gray-300 leading-relaxed">
              Transforma tus datos históricos de ventas en predicciones precisas. Utiliza inteligencia artificial para
              anticipar la demanda y tomar decisiones estratégicas.
            </p>

            <div className="flex flex-col sm:flex-row gap-4">
              <Link
                href="/upload"
                className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-lg font-bold text-lg flex items-center gap-2 justify-center"
              >
                Empieza a predecir tu futuro
                <ArrowRight className="h-5 w-5" />
              </Link>
            </div>

            <div className="flex items-center gap-8 text-gray-400">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-green-500" />
                <span>100% Gratuito</span>
              </div>
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-blue-500" />
                <span>Sin límites</span>
              </div>
              <div className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-purple-500" />
                <span>IA Avanzada</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-900 rounded-xl p-8 border border-gray-800">
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-white">Predicción de Ventas</h3>
                <div className="bg-green-600 text-white px-3 py-1 rounded-full text-sm font-bold">+24% proyectado</div>
              </div>

              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Enero 2024</span>
                  <span className="font-bold text-white">$45,230</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Febrero 2024</span>
                  <span className="font-bold text-white">$52,180</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Marzo 2024</span>
                  <span className="font-bold text-blue-400">$58,940</span>
                </div>
                <div className="border-t border-gray-700 pt-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 font-bold">Predicción Abril</span>
                    <span className="font-bold text-green-400">$64,120</span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-700 h-2 rounded-full">
                <div className="bg-blue-500 h-2 rounded-full w-3/4"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
