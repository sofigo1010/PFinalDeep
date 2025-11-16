import "./globals.css"

export const metadata = {
  title: "SalesVision AI - Predice tu futuro comercial",
  description: "Transforma tus datos históricos de ventas en predicciones precisas con inteligencia artificial",
}

export default function RootLayout({ children }) {
  return (
    <html lang="es">
      <body className="bg-black text-white">{children}</body>
    </html>
  )
}
