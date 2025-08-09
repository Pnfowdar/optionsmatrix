
self.addEventListener('install', e=>self.skipWaiting());
self.addEventListener('activate', e=>{});
self.addEventListener('fetch', ()=>{});
self.addEventListener('message', e=>{
  if(e.data?.type==='notify'){
    self.registration.showNotification(e.data.title,{body:e.data.body});
  }
});
