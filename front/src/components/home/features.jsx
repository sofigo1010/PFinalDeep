import { Upload, Brain, BarChart3, Target, Shield, Zap } from "lucide-react"

const features = [
  {
    icon: Upload,
    title: "Carga Fácil de Datos",
    description: "Sube tus archivos CSV de ventas históricas de forma simple y segura.",
  },
  {
    icon: Brain,
    title: "IA Avanzada",
    description: "Algoritmos de machine learning que analizan patrones complejos en tus datos.",
  },
  {
    icon: BarChart3,
    title: "Visualizaciones Intuitivas",
    description: "Gráficos y dashboards interactivos que hacen fácil entender tus predicciones.",
  },
  {
    icon: Target,
    title: "Predicciones Precisas",
    description: "Obtén predicciones de ventas futuras con alta precisión para planificar mejor.",
  },
  {
    icon: Shield,
    title: "Datos Seguros",
    description: "Tus datos están protegidos con encriptación de nivel empresarial.",
  },
  {
    icon: Zap,
    title: "Resultados Instantáneos",
    description: "Obtén predicciones en segundos, no en días. Toma decisiones rápidas.",
  },
]

export default function Features() {
  return (
    <section className="bg-gray-900 py-20">
      <div className="max-w-7xl mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">Todo lo que necesitas para predecir tus ventas</h2>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Herramientas poderosas y fáciles de usar que transforman tus datos históricos en insights accionables.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="bg-black border border-gray-800 rounded-xl p-8 hover:border-gray-600 transition-colors"
            >
              <div className="text-center space-y-4">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-blue-600 text-white mb-4">
                  <feature.icon className="h-8 w-8" />
                </div>
                <h3 className="text-xl font-bold text-white">{feature.title}</h3>
                <p className="text-gray-400 leading-relaxed">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
