import { Card, CardContent } from "../ui/card"
import { Upload, Settings, BarChart3, TrendingUp } from "lucide-react"

const steps = [
  {
    step: 1,
    icon: Upload,
    title: "Sube tus datos",
    description: "Carga tu archivo CSV con datos históricos de ventas. Nuestro sistema acepta múltiples formatos.",
    color: "bg-blue-500",
  },
  {
    step: 2,
    icon: Settings,
    title: "Procesamiento IA",
    description: "Nuestros algoritmos analizan automáticamente patrones, tendencias y estacionalidades en tus datos.",
    color: "bg-purple-500",
  },
  {
    step: 3,
    icon: BarChart3,
    title: "Visualiza resultados",
    description: "Explora dashboards interactivos con gráficos claros y métricas clave de rendimiento.",
    color: "bg-green-500",
  },
  {
    step: 4,
    icon: TrendingUp,
    title: "Toma decisiones",
    description: "Utiliza las predicciones para optimizar inventario, planificar campañas y crecer tu negocio.",
    color: "bg-orange-500",
  },
]

export default function HowItWorks() {
  return (
    <section className="py-20 bg-gray-50">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-900">Cómo funciona SalesVision AI</h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            En solo 4 pasos simples, transforma tus datos de ventas en predicciones precisas que impulsen el crecimiento
            de tu negocio.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {steps.map((step, index) => (
            <div key={index} className="relative">
              <Card className="border-0 shadow-lg hover:shadow-xl transition-all duration-300 h-full">
                <CardContent className="p-8 text-center space-y-6">
                  <div className="relative">
                    <div
                      className={`inline-flex items-center justify-center w-16 h-16 rounded-full ${step.color} text-white mb-4`}
                    >
                      <step.icon className="h-8 w-8" />
                    </div>
                    <div className="absolute -top-2 -right-2 bg-white border-2 border-gray-200 rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold text-gray-700">
                      {step.step}
                    </div>
                  </div>

                  <h3 className="text-xl font-semibold text-gray-900">{step.title}</h3>

                  <p className="text-gray-600 leading-relaxed">{step.description}</p>
                </CardContent>
              </Card>

              {index < steps.length - 1 && (
                <div className="hidden lg:block absolute top-1/2 -right-4 transform -translate-y-1/2 z-10">
                  <div className="w-8 h-0.5 bg-gray-300"></div>
                  <div className="absolute right-0 top-1/2 transform -translate-y-1/2 w-0 h-0 border-l-4 border-l-gray-300 border-t-2 border-b-2 border-t-transparent border-b-transparent"></div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
