import Link from "next/link"
import { TrendingUp } from "lucide-react"

export default function Footer() {
  return (
    <footer className="bg-gray-900 border-t border-gray-800">
      <div className="max-w-7xl mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <Link href="/" className="flex items-center gap-2 mb-4">
              <TrendingUp className="h-6 w-6 text-blue-500" />
              <span className="text-xl font-bold text-white">SalesVision AI</span>
            </Link>
            <p className="text-gray-400">Predice el futuro de tus ventas con inteligencia artificial.</p>
          </div>

          <div>
            <h3 className="font-bold text-white mb-4">Producto</h3>
            <div className="space-y-2">
              <Link href="/upload" className="block text-gray-400 hover:text-white">
                Subir Datos
              </Link>
              <Link href="/dashboard" className="block text-gray-400 hover:text-white">
                Dashboard
              </Link>
            </div>
          </div>

          <div>
            <h3 className="font-bold text-white mb-4">Recursos</h3>
            <div className="space-y-2">
              <Link href="/docs" className="block text-gray-400 hover:text-white">
                Documentación
              </Link>
              <Link href="/examples" className="block text-gray-400 hover:text-white">
                Ejemplos
              </Link>
            </div>
          </div>
        </div>

        <div className="border-t border-gray-800 mt-8 pt-8 text-center">
          <p className="text-gray-400">&copy; 2024 SalesVision AI. Todos los derechos reservados.</p>
        </div>
      </div>
    </footer>
  )
}
