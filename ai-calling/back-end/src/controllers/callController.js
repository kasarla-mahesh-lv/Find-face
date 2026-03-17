const callModel = require("../models/callModel")

async function incomingCall(request, reply) {

  return {
    message: "Welcome to our restaurant",
    options: [
      "Press 1 for order status",
      "Press 2 for customer support"
    ]
  }

}

async function ivrInput(request, reply) {

  const { phone, digit } = request.body

  await callModel.saveCall(phone, digit)

  if (digit == 1) {

    return { message: "Checking order status" }

  }

  if (digit == 2) {

    return { message: "Connecting to support team" }

  }

  return { message: "Invalid option" }

}

module.exports = {
  incomingCall,
  ivrInput
}