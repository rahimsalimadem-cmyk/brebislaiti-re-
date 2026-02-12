self.addEventListener('install', (e) => {
  console.log('Ovin Manager Service Worker installé');
});

self.addEventListener('fetch', (e) => {
  // Nécessaire pour valider le statut PWA
  return;
});
