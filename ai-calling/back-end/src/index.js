require('dotenv').config();
const fastify = require('fastify')({ logger: false });

fastify.register(require('@fastify/cors'),    { origin: '*' });
fastify.register(require('@fastify/helmet'));

fastify.get('/health', async () => ({
  status: 'ok',
  time: new Date().toISOString()
}));

fastify.register(require('./routes/calls'), { prefix: '/v1/calls' });

fastify.listen({ port: process.env.PORT || 3000, host: '0.0.0.0' }, (err) => {
  if (err) { console.error(err); process.exit(1); }
  console.log(`Server running on port ${process.env.PORT || 3000}`);
});
