 feature/dev-sravanig
 require("dotenv").config()

const fastify = require("fastify")({ logger: true })
const initDB = require("./src/utils/initDB")

async function start() {

  await initDB()

  fastify.get("/", async () => {
    return { message: "IVR backend running" }
  })

  await fastify.listen({ port: 3000 })

  console.log("Server running on port 3000")

}

start()
