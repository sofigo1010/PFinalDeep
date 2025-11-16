import Header from "../components/layout/header"
import Footer from "../components/layout/footer"
import Hero from "../components/home/hero"
import Features from "../components/home/features"
import CallToAction from "../components/home/call-to-action"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-black text-white">
      <Header />
      <main>
        <Hero />
        <Features />
        <CallToAction />
      </main>
      <Footer />
    </div>
  )
}
