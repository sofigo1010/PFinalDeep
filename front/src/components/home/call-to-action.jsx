import Link from "next/link"
import { ArrowRight, Sparkles } from "lucide-react"

export default function CallToAction() {
  return (
    <section className="bg-black py-20">
      <div className="max-w-7xl mx-auto px-4">
        <div className="text-center space-y-8">
          <div className="inline-flex items-center gap-2 bg-blue-600 rounded-full px-4 py-2 text-white">
            <Sparkles className="h-4 w-4" />
            <span className="font-bold">100% Gratuito • Sin límites</span>
          </div>

          <h2 className="text-4xl lg:text-5xl font-bold text-white leading-tight">
            ¿Listo para predecir el futuro de tus ventas?
          </h2>

          <p className="text-xl text-gray-400 leading-relaxed max-w-3xl mx-auto">
            Únete a miles de empresas que ya están utilizando SalesVision AI para tomar decisiones más inteligentes.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/upload"
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-lg font-bold text-lg flex items-center gap-2 justify-center"
            >
              Empieza a predecir tu futuro
              <ArrowRight className="h-5 w-5" />
            </Link>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-8 pt-8 border-t border-gray-800">
            <div className="text-center">
              <div className="text-3xl font-bold text-white">94%</div>
              <div className="text-gray-400">Precisión promedio</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-white">10K+</div>
              <div className="text-gray-400">Predicciones generadas</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-white">500+</div>
              <div className="text-gray-400">Empresas confiando</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
