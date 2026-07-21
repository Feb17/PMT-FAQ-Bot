import http from 'node:http';
import { RagClient } from './dist/ragClient.js';

const server = http.createServer((req, res) => {
  if (req.url === '/healthz') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end('{"ok":true}');
    return;
  }
  if (req.url === '/v1/chat/completions') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      const parsed = JSON.parse(body);
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ choices: [{ message: { content: `echo:${parsed.messages[0].content}` } }] }));
    });
    return;
  }
  res.writeHead(404); res.end();
});

server.listen(0, async () => {
  const { port } = server.address();
  const c = new RagClient(`http://127.0.0.1:${port}`, 'test', 1000);
  try {
    const health = await c.healthz();
    if (health !== true) throw new Error(`Expected health=true, got ${health}`);
    const answer = await c.chat('hello');
    if (answer.text !== 'echo:hello') throw new Error(`Expected echo answer, got ${JSON.stringify(answer)}`);
    console.log('smoke-rag-client: ok');
  } finally {
    server.close();
  }
});
