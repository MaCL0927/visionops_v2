const toast = document.getElementById('toast');
function showToast(message){toast.textContent=message;toast.classList.add('show');clearTimeout(showToast.t);showToast.t=setTimeout(()=>toast.classList.remove('show'),2600)}
const centerNotice=document.getElementById('centerNotice');
const centerNoticeBox=document.getElementById('centerNoticeBox');
function showCenterNotice(message,type='success',timeout=3600){
  if(!centerNotice || !centerNoticeBox){showToast(message);return;}
  centerNoticeBox.textContent=message;
  centerNoticeBox.className=`center-notice-box ${type}`;
  centerNotice.classList.add('show');
  clearTimeout(showCenterNotice.t);
  showCenterNotice.t=setTimeout(()=>centerNotice.classList.remove('show'),timeout);
}

async function apiGet(url){const res=await fetch(url);const data=await res.json();if(!res.ok) throw new Error(data.detail || data.message || '请求失败');return data}
async function apiPost(url, payload={}){const res=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await res.json();if(!res.ok) throw new Error(data.detail || data.message || '请求失败');return data}

const modeToggle=document.getElementById('modeToggle');
const factoryTabs=document.getElementById('factoryTabs');
const factoryMode=document.getElementById('factoryMode');
const productionMode=document.getElementById('productionMode');
const adminAuthModal=document.getElementById('adminAuthModal');
const adminUsername=document.getElementById('adminUsername');
const adminPassword=document.getElementById('adminPassword');
const ADMIN_USERNAME='admin';
const ADMIN_PASSWORD='admin';
let isProduction=false;
function enterProductionMode(){
  isProduction=true;
  stopRealtimeClassification();
  clearPendingClassificationCapture(false);
  modeToggle.textContent='切换工厂模式';
  modeToggle.classList.add('prod');
  factoryTabs.classList.add('hidden');
  factoryMode.classList.remove('active');
  productionMode.classList.add('active');
  updateCameraLifecycle();
}
function enterFactoryMode(){
  isProduction=false;
  modeToggle.textContent='切换生产模式';
  modeToggle.classList.remove('prod');
  factoryTabs.classList.remove('hidden');
  productionMode.classList.remove('active');
  factoryMode.classList.add('active');
  updateCameraLifecycle();
}
function openAdminAuthModal(){
  if(!adminAuthModal) return;
  adminPassword.value='';
  adminAuthModal.classList.add('active');
  setTimeout(()=>adminPassword && adminPassword.focus(),50);
}
function closeAdminAuthModal(){
  if(adminAuthModal) adminAuthModal.classList.remove('active');
}
modeToggle.addEventListener('click',()=>{
  if(isProduction){openAdminAuthModal();return;}
  enterProductionMode();
});
if(document.getElementById('closeAdminAuth')) document.getElementById('closeAdminAuth').addEventListener('click',closeAdminAuthModal);
if(document.getElementById('cancelAdminAuth')) document.getElementById('cancelAdminAuth').addEventListener('click',closeAdminAuthModal);
if(document.getElementById('confirmAdminAuth')) document.getElementById('confirmAdminAuth').addEventListener('click',()=>{
  const u=(adminUsername.value||'').trim();
  const p=adminPassword.value||'';
  if(u===ADMIN_USERNAME && p===ADMIN_PASSWORD){closeAdminAuthModal();enterFactoryMode();showToast('已切换到工厂模式');}
  else{showToast('管理员账号或密码错误');}
});
if(adminPassword) adminPassword.addEventListener('keydown',(e)=>{if(e.key==='Enter') document.getElementById('confirmAdminAuth').click();});

const topTabs=document.querySelectorAll('.top-tab');
const pages=document.querySelectorAll('.page');
topTabs.forEach(btn=>btn.addEventListener('click',()=>{
  if(btn.dataset.page!=='validatePage') stopRealtimeClassification();
  topTabs.forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  pages.forEach(p=>p.classList.remove('active'));
  document.getElementById(btn.dataset.page).classList.add('active');
  updateCameraLifecycle();
}));

const calibrationTips={positionCheck:'先完成摄像头位置校验，摄像头画面与半透明标准图对齐即可。',lightCheck:'再完成光照校验，观察左侧标准图与右侧实时画面的亮度和阴影是否接近。'};
const calStepButtons=document.querySelectorAll('[data-cal-step]');
const calScreens=[document.getElementById('positionCheck'),document.getElementById('lightCheck')];
calStepButtons.forEach(btn=>btn.addEventListener('click',()=>{calStepButtons.forEach(b=>b.classList.remove('active'));btn.classList.add('active');calScreens.forEach(s=>s.classList.remove('active'));document.getElementById(btn.dataset.calStep).classList.add('active');document.getElementById('calibrationTip').textContent=calibrationTips[btn.dataset.calStep];updateCameraLifecycle();}));

const collectButtons=document.querySelectorAll('[data-collect-step]');
const captureStep=document.getElementById('captureStep');
const uploadStep=document.getElementById('uploadStep');
collectButtons.forEach(btn=>btn.addEventListener('click',async()=>{
  collectButtons.forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  [captureStep,uploadStep].forEach(s=>s.classList.remove('active'));
  document.getElementById(btn.dataset.collectStep).classList.add('active');
  updateCameraLifecycle();
  if(btn.dataset.collectStep==='uploadStep') await refreshFolder(currentFolder);
}));

const cameraVideo=document.getElementById('cameraVideo');
const backendCamera=document.getElementById('backendCamera');
const captureCanvas=document.getElementById('captureCanvas');
const simulatedCamera=document.getElementById('simulatedCamera');
const cameraStatusTitle=document.getElementById('cameraStatusTitle');
const cameraStatusText=document.getElementById('cameraStatusText');
const captureTaskMode=document.getElementById('captureTaskMode');
const pendingCaptureModal=document.getElementById('pendingCaptureModal');
const pendingCaptureModalImage=document.getElementById('pendingCaptureModalImage');
const pendingCaptureZoomBox=document.getElementById('pendingCaptureZoomBox');
const savePositiveCaptureBtn=document.getElementById('savePositiveCapture');
const saveNegativeCaptureBtn=document.getElementById('saveNegativeCapture');
const cancelPendingCaptureBtn=document.getElementById('cancelPendingCapture');
const closePendingCaptureModalBtn=document.getElementById('closePendingCaptureModal');
const allCount=document.getElementById('allCount');
const positiveCount=document.getElementById('positiveCount');
const negativeCount=document.getElementById('negativeCount');
const allFolderCount=document.getElementById('allFolderCount');
const posFolderCount=document.getElementById('posFolderCount');
const negFolderCount=document.getElementById('negFolderCount');
const imageViewerModal=document.getElementById('imageViewerModal');
const viewerImage=document.getElementById('viewerImage');
const viewerTitle=document.getElementById('viewerTitle');
const viewerSubtitle=document.getElementById('viewerSubtitle');
let useRealCamera=false;
let useBackendCamera=false;
let backendCameraAvailable=false;
let cameraActive=false;
let dataStore={all:[],positive:[],negative:[]};
let currentDataset='local_dataset';
let currentFolder='all';
let viewerItem=null;
let deviceId='rk3588-001';
let userId='operator-001';
let selectedModel='';
let modelItems=[];
let pendingClassificationImageData='';
let pendingCaptureZoom=1;

function syncCounts(counts){
  const c=counts || {all:dataStore.all.length,positive:dataStore.positive.length,negative:dataStore.negative.length};
  allCount.textContent=c.all ?? 0; positiveCount.textContent=c.positive ?? 0; negativeCount.textContent=c.negative ?? 0;
  allFolderCount.textContent=`${c.all ?? 0} 张`; posFolderCount.textContent=`${c.positive ?? 0} 张`; negFolderCount.textContent=`${c.negative ?? 0} 张`;
}

async function loadCollectorState(){
  const data=await apiGet('/api/health');
  currentDataset=data.dataset || 'local_dataset';
  deviceId=data.device_id || deviceId;
  userId=data.user_id || userId;
  document.getElementById('uploadDeviceId').value=deviceId;
  syncCounts(data.counts);
  if(data.camera && data.camera.backend_enabled){
    backendCameraAvailable=true;
    useBackendCamera=true;
    useRealCamera=false;
  }
  await refreshFolder(currentFolder);
}

async function refreshDatasetSummary(){
  const data=await apiGet(`/api/dataset/summary?dataset=${encodeURIComponent(currentDataset)}`);
  deviceId=data.device_id || deviceId;
  userId=data.user_id || userId;
  document.getElementById('uploadDeviceId').value=deviceId;
  syncCounts(data.counts);
  await refreshFolder(currentFolder);
}

function isCapturePageVisible(){
  return !isProduction
    && factoryMode.classList.contains('active')
    && document.getElementById('collectPage').classList.contains('active')
    && captureStep.classList.contains('active');
}

function shouldKeepCameraRunning(){
  return isCapturePageVisible() || realtimeRunning;
}

function updateCameraLifecycle(){
  if(shouldKeepCameraRunning()) startCamera();
  else stopCamera();
}

async function startCamera(){
  if(cameraActive) return;
  cameraActive=true;
  // 部署到 RK3588 + 海康 RTSP 时，设置 VISIONOPS_CAMERA_SOURCE 后使用后端 MJPEG 流；
  // 普通笔记本调试时，仍优先使用浏览器 getUserMedia 读取电脑摄像头。
  if(backendCameraAvailable || useBackendCamera){
    useBackendCamera=true;
    backendCamera.src='/api/camera/stream?t=' + Date.now();
    backendCamera.classList.add('active','backend-active');
    cameraVideo.classList.remove('active');
    simulatedCamera.classList.remove('active');
    cameraStatusTitle.textContent='RTSP/后端摄像头画面';
    cameraStatusText.textContent='后端单例线程正在读取 RTSP，离开拍照采集页后会自动停止以降低资源占用';
    return;
  }
  if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){cameraStatusTitle.textContent='模拟摄像头画面';cameraStatusText.textContent='当前浏览器不支持摄像头接口，点击按钮后保存模拟图片';return;}
  try{
    const stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});
    cameraVideo.srcObject=stream;useRealCamera=true;useBackendCamera=false;simulatedCamera.classList.remove('active');backendCamera.classList.remove('active','backend-active');cameraVideo.classList.add('active');cameraStatusTitle.textContent='笔记本摄像头画面';cameraStatusText.textContent='点击上方按钮后，将当前帧直接保存到对应文件夹；离开本页后会自动关闭摄像头';
  }catch(err){useRealCamera=false;useBackendCamera=false;simulatedCamera.classList.add('active');backendCamera.classList.remove('active','backend-active');cameraStatusTitle.textContent='模拟摄像头画面';cameraStatusText.textContent='未获得摄像头权限，点击按钮后保存模拟图片';}
}

async function stopCamera(callBackend=true){
  if(cameraVideo && cameraVideo.srcObject){
    cameraVideo.srcObject.getTracks().forEach(track=>track.stop());
    cameraVideo.srcObject=null;
  }
  useRealCamera=false;
  if(backendCamera){
    backendCamera.removeAttribute('src');
    backendCamera.classList.remove('active','backend-active');
  }
  if(cameraVideo) cameraVideo.classList.remove('active');
  if(simulatedCamera) simulatedCamera.classList.add('active');
  cameraActive=false;
  if(cameraStatusTitle) cameraStatusTitle.textContent='摄像头已关闭';
  if(cameraStatusText) cameraStatusText.textContent='只有进入“采集上传 / 拍照采集”页面，或开启实时检测时才会打开摄像头';
  if(callBackend && (backendCameraAvailable || useBackendCamera)){
    try{await apiPost('/api/camera/stop');}catch(err){}
  }
}

function buildSimulatedFrame(ctx,w,h){
  const g=ctx.createLinearGradient(0,0,w,h); g.addColorStop(0,'#1e293b'); g.addColorStop(.55,'#475569'); g.addColorStop(1,'#020617');
  ctx.fillStyle=g; ctx.fillRect(0,0,w,h);
  ctx.fillStyle='rgba(255,255,255,.12)'; ctx.fillRect(w*.24,h*.28,w*.26,h*.28); ctx.fillRect(w*.60,h*.42,w*.23,h*.22);
  ctx.strokeStyle='#6ee7b7'; ctx.lineWidth=6; ctx.strokeRect(w*.24,h*.28,w*.26,h*.28);
  ctx.strokeStyle='#67e8f9'; ctx.strokeRect(w*.60,h*.42,w*.23,h*.22);
  ctx.fillStyle='white'; ctx.font='32px sans-serif'; ctx.fillText('SIMULATED CAMERA FRAME',32,58);
  ctx.font='24px sans-serif'; ctx.fillText(new Date().toLocaleString(),32,96);
}
function captureFrameAsJpeg(){
  const w=1280,h=720; captureCanvas.width=w; captureCanvas.height=h; const ctx=captureCanvas.getContext('2d');
  if(useBackendCamera && backendCamera.complete && backendCamera.naturalWidth>0){
    const vw=backendCamera.naturalWidth, vh=backendCamera.naturalHeight; const scale=Math.max(w/vw,h/vh); const sw=w/scale, sh=h/scale; const sx=(vw-sw)/2, sy=(vh-sh)/2; ctx.drawImage(backendCamera,sx,sy,sw,sh,0,0,w,h);
  } else if(useRealCamera && cameraVideo.videoWidth>0){
    const vw=cameraVideo.videoWidth, vh=cameraVideo.videoHeight; const scale=Math.max(w/vw,h/vh); const sw=w/scale, sh=h/scale; const sx=(vw-sw)/2, sy=(vh-sh)/2; ctx.drawImage(cameraVideo,sx,sy,sw,sh,0,0,w,h);
  } else {buildSimulatedFrame(ctx,w,h);}
  return captureCanvas.toDataURL('image/jpeg',0.88);
}
function resetPendingCaptureZoom(){
  pendingCaptureZoom=1;
  if(pendingCaptureModalImage){
    pendingCaptureModalImage.style.transform='scale(1)';
    pendingCaptureModalImage.style.transformOrigin='center center';
  }
}
function closePendingCaptureModal(clearData=false){
  if(pendingCaptureModal) pendingCaptureModal.classList.remove('active');
  resetPendingCaptureZoom();
  if(clearData) pendingClassificationImageData='';
}
function clearPendingClassificationCapture(showCamera=true){
  pendingClassificationImageData='';
  closePendingCaptureModal(false);
  if(showCamera){
    if(useBackendCamera && backendCamera) backendCamera.classList.add('active','backend-active');
    else if(useRealCamera && cameraVideo) cameraVideo.classList.add('active');
    else if(simulatedCamera) simulatedCamera.classList.add('active');
  }
}
function blobToDataUrl(blob){
  return new Promise((resolve,reject)=>{
    const reader=new FileReader();
    reader.onload=()=>resolve(reader.result);
    reader.onerror=reject;
    reader.readAsDataURL(blob);
  });
}
async function captureCurrentFrameDataUrl(){
  await startCamera();
  await new Promise(resolve=>setTimeout(resolve,160));
  if(useBackendCamera){
    const res=await fetch('/api/camera/frame?t='+Date.now());
    if(!res.ok) throw new Error('读取摄像头当前帧失败');
    return await blobToDataUrl(await res.blob());
  }
  return captureFrameAsJpeg();
}
function showPendingClassificationCapture(imageData){
  pendingClassificationImageData=imageData;
  // 主摄像头画面保持实时预览，不再被暂存图替换。暂存图只在弹窗中展示。
  if(pendingCaptureModalImage){
    pendingCaptureModalImage.src=imageData;
  }
  resetPendingCaptureZoom();
  if(pendingCaptureModal) pendingCaptureModal.classList.add('active');
  if(cameraStatusTitle) cameraStatusTitle.textContent='实时摄像头画面';
  if(cameraStatusText) cameraStatusText.textContent='图片已暂存到弹窗中，请在弹窗里放大查看并选择合格/不合格，或取消放弃';
}
async function captureToFolder(folder){
  try{
    const payload={dataset:currentDataset,folder,device_id:deviceId,user_id:userId};
    // v4.7：RTSP/后端摄像头模式下，后端直接保存 latest_frame；
    // 笔记本浏览器摄像头/模拟画面模式下，仍由前端 canvas 截图上传。
    if(!useBackendCamera){
      payload.image_data=captureFrameAsJpeg();
    }
    const data=await apiPost('/api/capture',payload);
    syncCounts(data.counts);
    await refreshFolder(currentFolder);
    showToast(data.message);
  }catch(err){showToast(err.message)}
}
async function handleCaptureButton(){
  const mode=(captureTaskMode && captureTaskMode.value) || 'classification';
  clearPendingClassificationCapture(false);
  if(mode==='detection'){
    await captureToFolder('all');
    return;
  }
  try{
    const imageData=await captureCurrentFrameDataUrl();
    showPendingClassificationCapture(imageData);
    showToast('图片已暂存，请选择合格或不合格');
  }catch(err){showToast(err.message || '取图失败')}
}
async function savePendingClassificationCapture(label){
  if(!pendingClassificationImageData){showToast('没有暂存图片，请先取图');return;}
  try{
    const data=await apiPost('/api/capture/labeled',{dataset:currentDataset,image_data:pendingClassificationImageData,label,device_id:deviceId,user_id:userId});
    syncCounts(data.counts);
    clearPendingClassificationCapture(true);
    await refreshFolder(label);
    showToast(data.message);
  }catch(err){showToast(err.message || '保存失败')}
}

document.getElementById('captureAllBtn').addEventListener('click',handleCaptureButton);
if(savePositiveCaptureBtn) savePositiveCaptureBtn.addEventListener('click',()=>savePendingClassificationCapture('positive'));
if(saveNegativeCaptureBtn) saveNegativeCaptureBtn.addEventListener('click',()=>savePendingClassificationCapture('negative'));
if(cancelPendingCaptureBtn) cancelPendingCaptureBtn.addEventListener('click',()=>{clearPendingClassificationCapture(true);showToast('已取消暂存图片')});
if(closePendingCaptureModalBtn) closePendingCaptureModalBtn.addEventListener('click',()=>{clearPendingClassificationCapture(true);showToast('已取消暂存图片')});
if(pendingCaptureZoomBox){
  pendingCaptureZoomBox.addEventListener('wheel',(e)=>{
    if(!pendingClassificationImageData) return;
    e.preventDefault();
    const delta=e.deltaY<0 ? 0.12 : -0.12;
    pendingCaptureZoom=Math.min(5, Math.max(1, pendingCaptureZoom+delta));
    if(pendingCaptureModalImage){
      pendingCaptureModalImage.style.transform=`scale(${pendingCaptureZoom})`;
    }
  }, {passive:false});
}

const folderCards=document.querySelectorAll('.folder-card');
const previewGrid=document.getElementById('folderPreviewGrid');
function folderLabel(folder){return folder==='all'?'检测图片':folder==='positive'?'合格':'不合格'}
async function refreshFolder(folder='all'){
  currentFolder=folder;
  folderCards.forEach(b=>b.classList.toggle('active',b.dataset.folderCard===folder));
  try{
    const data=await apiGet(`/api/dataset/images?dataset=${encodeURIComponent(currentDataset)}&folder=${encodeURIComponent(folder)}`);
    dataStore[folder]=data.items; syncCounts(data.counts); renderFolder(folder,data.items);
  }catch(err){showToast(err.message)}
}
function renderFolder(folder,items){
  previewGrid.innerHTML='';
  if(!items || items.length===0){previewGrid.innerHTML=`<div class="empty-preview">${folderLabel(folder)} 文件夹暂无图片</div>`;return;}
  items.slice(0,120).forEach((item)=>{
    const card=document.createElement('div'); card.className='preview-card';
    card.innerHTML=`<div class="preview-thumb-wrap"><img class="preview-img" src="${item.url}?t=${Date.now()}" alt="${item.filename}"><button class="delete-chip" title="删除图片">×</button></div><div class="preview-meta"><b title="${item.filename}">${item.filename}</b><span>${folderLabel(folder)} · ${item.mtime}</span></div>`;
    card.querySelector('.preview-img').addEventListener('click',()=>openImageViewer(item));
    card.querySelector('.delete-chip').addEventListener('click',(e)=>{e.stopPropagation();deletePreviewImage(item)});
    previewGrid.appendChild(card);
  });
}
folderCards.forEach(btn=>btn.addEventListener('click',()=>refreshFolder(btn.dataset.folderCard)));
if(captureTaskMode) captureTaskMode.addEventListener('change',()=>{clearPendingClassificationCapture(true);showToast(captureTaskMode.value==='classification'?'已切换到分类采集模式':'已切换到检测采集模式')});

function openImageViewer(item){
  viewerItem=item;
  document.getElementById('deleteViewerImage').style.display='';
  viewerImage.src=item.url + `?t=${Date.now()}`;
  const deleteBtn=document.getElementById('deleteViewerImage'); if(deleteBtn) deleteBtn.style.display='';
  viewerTitle.textContent=item.filename;
  viewerSubtitle.textContent=`${folderLabel(item.folder)}。删除规则：在检测图片中删除会同步移除合格/不合格同名副本；在合格/不合格中删除只移除该标签图片。`;
  imageViewerModal.classList.add('active');
}
document.getElementById('closeImageViewer').addEventListener('click',()=>imageViewerModal.classList.remove('active'));
document.getElementById('deleteViewerImage').addEventListener('click',async()=>{if(viewerItem) await deletePreviewImage(viewerItem,true)});
async function deletePreviewImage(item,fromViewer=false){
  if(!item) return;
  const ok=window.confirm(`确认删除 ${item.filename} 吗？`);
  if(!ok) return;
  try{
    const data=await apiPost('/api/dataset/image/delete',{dataset:currentDataset,folder:item.folder,filename:item.filename});
    syncCounts(data.counts);
    if(fromViewer) imageViewerModal.classList.remove('active');
    await refreshFolder(currentFolder);
    showToast('图片已删除');
  }catch(err){showToast(err.message)}
}

const uploadModal=document.getElementById('uploadModal');

const clearCaptureImagesBtn=document.getElementById('clearCaptureImages');
if(clearCaptureImagesBtn){
  clearCaptureImagesBtn.addEventListener('click',async()=>{
    const ok=window.confirm('确认清空检测图片、合格、不合格文件夹下的所有图片吗？此操作不可恢复。');
    if(!ok) return;
    try{
      const data=await apiPost('/api/dataset/images/clear',{dataset:currentDataset});
      syncCounts(data.counts);
      dataStore={all:[],positive:[],negative:[]};
      await refreshFolder(currentFolder);
      await loadValidationImages();
      showCenterNotice(data.message || '已清空采集图片','success',3600);
    }catch(err){
      showCenterNotice(err.message || '清空失败','error',4600);
    }
  });
}
document.getElementById('openUploadModal').addEventListener('click',async()=>{await refreshFolder(currentFolder);uploadModal.classList.add('active')});
document.getElementById('closeUploadModal').addEventListener('click',()=>uploadModal.classList.remove('active'));
document.getElementById('cancelUploadModal').addEventListener('click',()=>uploadModal.classList.remove('active'));
document.getElementById('confirmUpload').addEventListener('click',async()=>{
  const payload={dataset:currentDataset,device_id:document.getElementById('uploadDeviceId').value.trim(),customer_id:document.getElementById('uploadCustomerId').value.trim(),contact_info:document.getElementById('uploadContact').value.trim(),remark:document.getElementById('uploadRemark').value.trim()};
  try{const data=await apiPost('/api/upload',payload);const remote=data.package&&data.package.remote_upload;const suffix=remote&&remote.uploaded?` → ${remote.target}`:`：${data.package.package}`;showCenterNotice(`${data.message}${suffix}`,'success',4200);uploadModal.classList.remove('active');await refreshDatasetSummary();}
  catch(err){showCenterNotice(err.message || '上传失败，请检查网络或目标电脑配置','error',5200)}
});

function escapeHtml(text){
  return String(text ?? '').replace(/[&<>'"]/g,(ch)=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
}

function normalizeTaskName(task){
  const t=String(task || '').trim().toLowerCase();
  if(['obb','obb_detection','oriented_detection','oriented_bbox_detection','rotated_detection','rotated_bbox_detection','yolo_obb','yolov8_obb'].includes(t)) return 'obb_detection';
  if(['detect','detection','yolo_detection','object_detection'].includes(t)) return 'detection';
  if(['cls','classify','classification','image_classification'].includes(t)) return 'classification';
  return t;
}
function isDetectionLikeTask(task){
  const t=normalizeTaskName(task);
  return t==='detection' || t==='obb_detection';
}
function taskDisplayName(task){
  const t=normalizeTaskName(task);
  if(t==='classification') return '分类';
  if(t==='detection') return '检测';
  if(t==='obb_detection') return '旋转框检测';
  return task || '未知';
}


function formatBBox(bbox){
  if(!Array.isArray(bbox) || bbox.length<4) return '--';
  return `[${bbox.slice(0,4).map((x)=>Number(x).toFixed(2)).join(', ')}]`;
}
function formatPoint(point){
  if(!Array.isArray(point) || point.length<2) return '--';
  return `[${Number(point[0]).toFixed(2)}, ${Number(point[1]).toFixed(2)}]`;
}
function getPredictionCenter(pred, fallbackBox=null){
  if(Array.isArray(pred.center) && pred.center.length>=2){
    return [Number(pred.center[0]), Number(pred.center[1])];
  }
  if(Number.isFinite(Number(pred.center_x)) && Number.isFinite(Number(pred.center_y))){
    return [Number(pred.center_x), Number(pred.center_y)];
  }
  const box = fallbackBox || (Array.isArray(pred.bbox) ? pred.bbox.map(Number) : null);
  if(box && box.length>=4 && !box.some((x)=>!Number.isFinite(x))){
    return [(box[0]+box[2])/2, (box[1]+box[3])/2];
  }
  return null;
}

let validationImages=[];
let selectedImage=null;

const selectedModelNameEl=document.getElementById('selectedModelName');
const selectedImageNameEl=document.getElementById('selectedImageName');
const validationImageGrid=document.getElementById('validationImageGrid');
const selectedImagePreview=document.getElementById('selectedImagePreview');
const classificationResult=document.getElementById('classificationResult');
const resultClassName=document.getElementById('resultClassName');
const resultConfidence=document.getElementById('resultConfidence');
const resultLatency=document.getElementById('resultLatency');
const topkResult=document.getElementById('topkResult');
const realtimeInferToggle=document.getElementById('realtimeInferToggle');
const REALTIME_INTERVAL_MS=1000; // v6.7: 分类/检测统一使用低频单帧实时推理
let realtimeRunning=false;
let realtimeTimer=null;
let realtimeBusy=false;

function updateSelectedModelUI(){
  if(!selectedModelNameEl) return;
  const item=getSelectedModelItem ? getSelectedModelItem() : null;
  if(!selectedModel){selectedModelNameEl.textContent='未找到模型';return;}
  if(item && item.has_meta!==false){
    const taskText=item.task_label || item.task || '未知任务';
    const clsText=item.num_classes ? ` · ${item.num_classes}类` : '';
    selectedModelNameEl.textContent=`${selectedModel}（${taskText}${clsText}）`;
  }else{
    selectedModelNameEl.textContent=`${selectedModel}（配置缺失）`;
  }
}
function setResultIdle(message='等待检测'){
  if(!classificationResult) return;
  classificationResult.className='classification-result idle';
  resultClassName.textContent=message;
  resultConfidence.textContent='结果 --';
  resultLatency.textContent='耗时 --';
  if(topkResult) topkResult.innerHTML='';
  window.__lastDetectionPredictions=[];
  clearDetectionOverlay();
}
function getSelectedModelItem(){
  return modelItems.find((item)=>item.name===selectedModel) || null;
}
function renderModelList(){
  const modelList=document.getElementById('modelList');
  if(!modelList) return;
  if(!modelItems.length){
    selectedModel='';
    updateSelectedModelUI();
    modelList.innerHTML='<div class="empty-models">未找到模型<br><small>/opt/visionops/models 下暂无 .rknn 文件</small></div>';
    return;
  }
  modelList.innerHTML=modelItems.map((item)=>{
    const active=item.name===selectedModel?'active':'';
    const sizeText=Number(item.size_mb)>0?`${item.size_mb} MB`:'';
    const taskText=item.has_meta ? (item.task_label || item.task || '未知任务') : '配置缺失';
    const classText=item.has_meta && item.num_classes ? `${item.num_classes}类` : '';
    const customerText=item.customer_id ? item.customer_id : '';
    const sub=[item.label || taskText, classText, customerText, sizeText].filter(Boolean).join(' · ');
    const disabled=item.has_meta===false ? ' data-missing-meta="1"' : '';
    return `<button class="model-card ${active}" data-model-name="${escapeHtml(item.name)}" title="${escapeHtml(item.meta_path || item.path || item.name)}"${disabled}><b>${escapeHtml(item.name)}</b><span>${escapeHtml(sub)}</span></button>`;
  }).join('');
  modelList.querySelectorAll('.model-card').forEach((btn)=>{
    btn.addEventListener('click',()=>{
      stopRealtimeClassification();
      selectedModel=btn.dataset.modelName || '';
      renderModelList();
      const item=getSelectedModelItem();
      if(item && item.has_meta===false){
        setResultIdle('模型配置缺失');
        showToast('该模型缺少同名 yaml，无法验证');
      }else{
        setResultIdle('等待检测');
        showToast(selectedModel ? `已选择模型：${selectedModel}` : '请选择模型');
      }
    });
  });
  updateSelectedModelUI();
}
function pickDefaultModel(){
  const usable=modelItems.find((item)=>item.has_meta!==false);
  selectedModel=usable ? usable.name : (modelItems[0] ? modelItems[0].name : '');
  updateSelectedModelUI();
}
async function loadModels(){
  const modelList=document.getElementById('modelList');
  if(!modelList) return;
  modelList.innerHTML='<div class="empty-models">正在读取模型...</div>';
  try{
    const data=await apiGet('/api/models');
    modelItems=data.items || [];
    pickDefaultModel();
    renderModelList();
  }catch(err){
    modelList.innerHTML=`<div class="empty-models">模型读取失败<br><small>${escapeHtml(err.message || err)}</small></div>`;
  }
}

function setSelectedValidationImage(item, resetResult=true){
  selectedImage=item || null;
  window.__lastDetectionPredictions=[];
  if(selectedImageNameEl) selectedImageNameEl.textContent=selectedImage ? selectedImage.name : '请先选择图片';
  if(selectedImage){
    renderPreviewImage(`${selectedImage.url}?t=${Date.now()}`, selectedImage.name, []);
  }else if(selectedImagePreview){
    selectedImagePreview.innerHTML='<span>选择左侧图片，或点击“拍照检测 / 实时检测”后在这里预览</span>';
  }
  if(resetResult) setResultIdle('等待检测');
}

function renderValidationImages(){
  if(!validationImageGrid) return;
  if(!validationImages.length){
    validationImageGrid.innerHTML='<div class="empty-models">暂无采集图片<br><small>请先到采集上传页取图，或直接点击拍照检测</small></div>';
    setSelectedValidationImage(null, false);
    return;
  }
  validationImageGrid.innerHTML=validationImages.map((item)=>{
    const active=selectedImage && selectedImage.id===item.id ? 'active' : '';
    return `<button class="validation-image-card ${active}" data-image-id="${escapeHtml(item.id)}"><img src="${item.url}?t=${Date.now()}" alt="${escapeHtml(item.name)}"><span title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span></button>`;
  }).join('');
  validationImageGrid.querySelectorAll('.validation-image-card').forEach((btn)=>{
    btn.addEventListener('click',()=>{
      const item=validationImages.find((x)=>x.id===btn.dataset.imageId) || null;
      setSelectedValidationImage(item);
      renderValidationImages();
    });
  });
}
async function loadValidationImages(){
  if(!validationImageGrid) return;
  validationImageGrid.innerHTML='<div class="empty-models">正在读取图片...</div>';
  try{
    const data=await apiGet(`/api/validation/images?dataset=${encodeURIComponent(currentDataset)}`);
    validationImages=data.items || [];
    if(!selectedImage && validationImages.length){
      setSelectedValidationImage(validationImages[0], false);
    }
    renderValidationImages();
  }catch(err){
    validationImageGrid.innerHTML=`<div class="empty-models">图片读取失败<br><small>${escapeHtml(err.message || err)}</small></div>`;
  }
}

function renderPreviewImage(url, alt='测试图片', detections=[]){
  if(!selectedImagePreview) return;
  selectedImagePreview.innerHTML=`
    <div class="preview-canvas-wrap">
      <img id="resultPreviewImg" src="${url}" alt="${escapeHtml(alt)}">
      <canvas id="detectionOverlayCanvas" aria-hidden="true"></canvas>
    </div>
  `;
  const img=document.getElementById('resultPreviewImg');
  if(!img) return;
  img.addEventListener('load',()=>drawDetectionOverlay(detections));
  if(img.complete && img.naturalWidth>0){
    requestAnimationFrame(()=>drawDetectionOverlay(detections));
  }
}
function clearDetectionOverlay(){
  const canvas=document.getElementById('detectionOverlayCanvas');
  if(!canvas) return;
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  canvas.style.display='none';
}
function getPredictionLabelPosition(pred){
  const points=pred && pred.obb && Array.isArray(pred.obb.points) ? pred.obb.points : null;
  if(points && points.length){
    return points.reduce((best,p)=>{
      const x=Number(p && p[0]);
      const y=Number(p && p[1]);
      if(!Number.isFinite(x) || !Number.isFinite(y)) return best;
      if(!best) return [x,y];
      if(y<best[1] || (y===best[1] && x<best[0])) return [x,y];
      return best;
    }, null) || [0,0];
  }
  const box=Array.isArray(pred && pred.bbox) ? pred.bbox.map(Number) : null;
  if(box && box.length>=2 && Number.isFinite(box[0]) && Number.isFinite(box[1])) return [box[0],box[1]];
  return [0,0];
}
function getValidObbPoints(pred, canvasWidth, canvasHeight){
  const points=pred && pred.obb && Array.isArray(pred.obb.points) ? pred.obb.points : null;
  if(!points || points.length<4) return null;
  const normalized=[];
  for(const p of points){
    if(!Array.isArray(p) || p.length<2) return null;
    const x=Number(p[0]);
    const y=Number(p[1]);
    if(!Number.isFinite(x) || !Number.isFinite(y)) return null;
    normalized.push([
      Math.max(0, Math.min(canvasWidth, x)),
      Math.max(0, Math.min(canvasHeight, y)),
    ]);
  }
  return normalized;
}
function drawPolygonOverlay(ctx, points){
  if(!points || points.length<4) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for(let i=1;i<points.length;i++) ctx.lineTo(points[i][0], points[i][1]);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}
function drawDetectionOverlay(predictions=[]){
  const canvas=document.getElementById('detectionOverlayCanvas');
  const img=document.getElementById('resultPreviewImg');
  const wrap=selectedImagePreview ? selectedImagePreview.querySelector('.preview-canvas-wrap') : null;
  if(!canvas || !img || !wrap) return;
  if(!Array.isArray(predictions) || predictions.length===0 || !img.naturalWidth || !img.naturalHeight){
    clearDetectionOverlay();
    return;
  }
  canvas.width=img.naturalWidth;
  canvas.height=img.naturalHeight;
  const imgRect=img.getBoundingClientRect();
  const wrapRect=wrap.getBoundingClientRect();
  canvas.style.display='block';
  canvas.style.left=`${imgRect.left-wrapRect.left}px`;
  canvas.style.top=`${imgRect.top-wrapRect.top}px`;
  canvas.style.width=`${imgRect.width}px`;
  canvas.style.height=`${imgRect.height}px`;
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.lineWidth=Math.max(3, Math.round(canvas.width/360));
  ctx.font=`${Math.max(18, Math.round(canvas.width/48))}px sans-serif`;
  ctx.textBaseline='top';
  predictions.forEach((pred)=>{
    const obbPoints=getValidObbPoints(pred, canvas.width, canvas.height);
    const box=Array.isArray(pred.bbox) ? pred.bbox.map(Number) : null;

    let labelAnchor=null;
    let hasShape=false;
    ctx.strokeStyle=obbPoints ? '#06b6d4' : '#22c55e';
    ctx.fillStyle=obbPoints ? 'rgba(6,182,212,.16)' : 'rgba(34,197,94,.18)';

    if(obbPoints){
      drawPolygonOverlay(ctx, obbPoints);
      labelAnchor=getPredictionLabelPosition(pred);
      hasShape=true;
    }else if(box && box.length>=4 && !box.some((x)=>!Number.isFinite(x))){
      const x1=Math.max(0, Math.min(canvas.width, box[0]));
      const y1=Math.max(0, Math.min(canvas.height, box[1]));
      const x2=Math.max(0, Math.min(canvas.width, box[2]));
      const y2=Math.max(0, Math.min(canvas.height, box[3]));
      const w=Math.max(0, x2-x1);
      const h=Math.max(0, y2-y1);
      if(w>=2 && h>=2){
        ctx.fillRect(x1,y1,w,h);
        ctx.strokeRect(x1,y1,w,h);
        labelAnchor=[x1,y1];
        hasShape=true;
      }
    }
    if(!hasShape) return;

    const cls=String(pred.class_name ?? pred.class ?? pred.label ?? pred.class_id ?? '目标');
    const conf=Number(pred.confidence ?? pred.score);
    const label=Number.isFinite(conf) ? `${cls} ${(conf*100).toFixed(1)}%` : cls;
    const pad=Math.max(6, Math.round(canvas.width/180));
    const textW=ctx.measureText(label).width;
    const labelH=Math.max(24, Math.round(canvas.width/36));
    const anchorX=Number(labelAnchor && labelAnchor[0]);
    const anchorY=Number(labelAnchor && labelAnchor[1]);
    const lx=Math.max(0, Math.min(canvas.width-(textW+pad*2), Number.isFinite(anchorX) ? anchorX : 0));
    const ly=Math.max(0, (Number.isFinite(anchorY) ? anchorY : 0)-labelH-2);
    ctx.fillStyle='rgba(15,23,42,.90)';
    ctx.fillRect(lx,ly,textW+pad*2,labelH);
    ctx.fillStyle='#fff';
    ctx.fillText(label,lx+pad,ly+pad/2);

    const center=getPredictionCenter(pred, box);
    if(center && Number.isFinite(center[0]) && Number.isFinite(center[1])){
      const cx=Math.max(0, Math.min(canvas.width, center[0]));
      const cy=Math.max(0, Math.min(canvas.height, center[1]));
      const r=Math.max(4, Math.round(canvas.width/180));
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI*2);
      ctx.fillStyle='#ef4444';
      ctx.fill();
      ctx.lineWidth=Math.max(2, Math.round(canvas.width/500));
      ctx.strokeStyle='#fff';
      ctx.stroke();

      const centerText=`(${cx.toFixed(1)}, ${cy.toFixed(1)})`;
      ctx.font=`${Math.max(14, Math.round(canvas.width/58))}px sans-serif`;
      const cw=ctx.measureText(centerText).width;
      const ch=Math.max(22, Math.round(canvas.width/44));
      const tx=Math.min(canvas.width-cw-pad*2, cx+r+6);
      const ty=Math.min(canvas.height-ch, cy+r+6);
      ctx.fillStyle='rgba(15,23,42,.90)';
      ctx.fillRect(Math.max(0,tx), Math.max(0,ty), cw+pad*2, ch);
      ctx.fillStyle='#fff';
      ctx.fillText(centerText, Math.max(0,tx)+pad, Math.max(0,ty)+pad/2);
    }
  });
}
window.addEventListener('resize',()=>{
  const last=window.__lastDetectionPredictions || [];
  if(last.length) drawDetectionOverlay(last);
});

function setResultRunning(message='检测中...'){
  classificationResult.className='classification-result running';
  resultClassName.textContent=message;
  resultConfidence.textContent='正在加载模型并推理';
  resultLatency.textContent='请稍候';
  if(topkResult) topkResult.innerHTML='';
}
function renderClassificationResult(data, options={}){
  const task=data.task || 'classification';
  const r=data.result || {};
  classificationResult.className='classification-result done';
  if(isDetectionLikeTask(task)){
    const predictions=Array.isArray(data.predictions) ? data.predictions : [];
    window.__lastDetectionPredictions=predictions;
    const taskText=normalizeTaskName(task)==='obb_detection' ? '旋转框检测' : '检测';
    const updatedAt=data.updated_at || new Date().toLocaleTimeString();
    resultClassName.textContent=`耗时 ${data.latency_ms ?? '--'} ms`;
    resultConfidence.textContent=`${taskText} · 共 ${predictions.length} 个目标`;
    resultLatency.textContent=`更新 ${updatedAt}`;

    if(topkResult){
      if(predictions.length){
        topkResult.innerHTML='<b>目标明细</b>'+predictions.map((p,idx)=>{
          const bbox=Array.isArray(p.bbox) ? p.bbox : [];
          const center=getPredictionCenter(p);
          const conf=Number(p.confidence ?? p.score);
          const confText=Number.isFinite(conf) ? `${(conf*100).toFixed(1)}%` : '--';
          const cls=escapeHtml(p.class_name ?? p.class ?? p.label ?? p.class_id ?? '目标');
          return `<div class="target-row target-row-inline">
            <span class="target-index">目标${idx+1}</span>
            <span class="target-class">${cls}</span>
            <span>位置：${escapeHtml(formatBBox(bbox))}</span>
            <span>中心点：${escapeHtml(formatPoint(center))}</span>
            <span>置信度：${confText}</span>
            <span>更新时间：${escapeHtml(updatedAt)}</span>
          </div>`;
        }).join('');
      }else{
        topkResult.innerHTML='<b>目标明细</b><div class="target-row empty">未检测到目标</div>';
      }
    }
    drawDetectionOverlay(predictions);
    if(!options.silent) showToast(`${taskText}完成：${predictions.length} 个目标`);
    return;
  }
  window.__lastDetectionPredictions=[];
  clearDetectionOverlay();
  resultClassName.textContent=r.class_name || '未识别';
  resultConfidence.textContent=`置信度 ${r.confidence_percent || '--'}`;
  resultLatency.textContent=`耗时 ${data.latency_ms ?? '--'} ms`;
  if(Array.isArray(data.topk) && data.topk.length && topkResult){
    topkResult.innerHTML='<b>Top 结果</b>'+data.topk.map((x)=>{
      const conf=Number(x.confidence ?? x.score);
      const confText=Number.isFinite(conf) ? (conf*100).toFixed(1)+'%' : '--';
      return `<span>${escapeHtml(x.class_name ?? x.class ?? x.label ?? x.class_id)}：${confText}</span>`;
    }).join('');
  }else if(topkResult){
    topkResult.innerHTML='';
  }
  if(!options.silent) showToast(`检测完成：${r.class_name || '未识别'} ${r.confidence_percent || ''}`);
}
function renderClassificationError(err){
  classificationResult.className='classification-result error';
  resultClassName.textContent='检测失败';
  resultConfidence.textContent=err.message || '请检查模型或图片';
  resultLatency.textContent='耗时 --';
  showToast(err.message || '检测失败');
}
async function runSingleImageClassification(){
  if(!selectedModel){showToast('未找到模型，请先刷新模型列表');return;}
  const item=getSelectedModelItem();
  if(item && item.has_meta===false){showToast('该模型缺少同名 yaml，无法验证');return;}
  if(!selectedImage){showToast('请先选择一张测试图片，或点击拍照检测');return;}
  setResultRunning('检测中...');
  try{
    const data=await apiPost('/api/validation/classify_image',{dataset:currentDataset,model_name:selectedModel,image_id:selectedImage.id});
    renderClassificationResult(data,{mode:'image'});
  }catch(err){
    renderClassificationError(err);
  }
}

async function captureAndRunClassification(){
  // v6.7：分类/检测模型都复用该接口；检测模型会在结果图上叠加检测框。
  if(!selectedModel){showToast('未找到模型，请先刷新模型列表');return;}
  const item=getSelectedModelItem();
  if(item && item.has_meta===false){showToast('该模型缺少同名 yaml，无法验证');return;}
  setResultRunning('拍照检测中...');
  try{
    await startCamera();
    await new Promise(resolve=>setTimeout(resolve, 250));
    const payload={dataset:currentDataset,model_name:selectedModel,device_id:deviceId,user_id:userId};
    if(!useBackendCamera){
      payload.image_data=captureFrameAsJpeg();
    }
    const data=await apiPost('/api/validation/capture_classify',payload);
    if(data.captured){
      const captured={...data.captured, url:data.captured.url || `/api/validation/image/${encodeURIComponent(data.captured.id)}`};
      validationImages=[captured, ...validationImages.filter((item)=>item.id!==captured.id)];
      setSelectedValidationImage(captured, false);
      renderValidationImages();
      await refreshDatasetSummary();
    }
    renderClassificationResult(data,{mode:'image'});
  }catch(err){
    renderClassificationError(err);
  }finally{
    if(!shouldKeepCameraRunning()) stopCamera();
  }
}
document.getElementById('refreshModels').addEventListener('click',async()=>{
  stopRealtimeClassification();
  try{
    const data=await apiPost('/api/refresh_models');
    modelItems=data.items || [];
    pickDefaultModel();
    renderModelList();
    setResultIdle('等待检测');
    showToast(data.message || `已刷新模型列表，共找到 ${modelItems.length} 个模型`);
  }catch(err){showToast(err.message || '刷新模型失败')}
});
document.getElementById('refreshValidationImages').addEventListener('click',async()=>{selectedImage=null;await loadValidationImages();showToast('已刷新测试图片')});
document.getElementById('runSingleImageInfer').addEventListener('click',runSingleImageClassification);
document.getElementById('captureAndInfer').addEventListener('click',captureAndRunClassification);
if(realtimeInferToggle){realtimeInferToggle.addEventListener('click',toggleRealtimeInferUI);}


function setRealtimeButtonState(running){
  if(!realtimeInferToggle) return;
  realtimeInferToggle.classList.toggle('active', running);
  realtimeInferToggle.setAttribute('aria-pressed', running ? 'true' : 'false');
  realtimeInferToggle.textContent=running ? '停止实时' : '实时检测';
}

async function realtimeClassifyOnce(){
  // v6.7：检测模型实时检测，前端按返回 bbox 持续更新叠加框。
  if(!realtimeRunning || realtimeBusy) return;
  if(!selectedModel){stopRealtimeClassification();showToast('未找到模型，请先刷新模型列表');return;}
  realtimeBusy=true;
  try{
    const payload={dataset:currentDataset,model_name:selectedModel};
    // RTSP/后端摄像头模式下，后端直接使用 latest_frame；浏览器摄像头/模拟模式上传当前 canvas 截图。
    if(!useBackendCamera){
      payload.image_data=captureFrameAsJpeg();
    }
    const data=await apiPost('/api/validation/realtime_classify_once',payload);
    if(data.realtime){
      const url=data.realtime.url || '/api/validation/realtime_image/realtime_latest.jpg?t='+Date.now();
      renderPreviewImage(url, '实时检测画面', data.predictions || []);
      if(selectedImageNameEl){
        selectedImageNameEl.textContent='实时画面';
      }
    }
    renderClassificationResult(data,{silent:true,mode:'realtime'});
    if(resultLatency){
      const now=new Date().toLocaleTimeString();
      resultLatency.textContent=`耗时 ${data.latency_ms ?? '--'} ms · 更新 ${now}`;
    }
  }catch(err){
    renderClassificationError(err);
    stopRealtimeClassification(false);
  }finally{
    realtimeBusy=false;
  }
}

async function startRealtimeClassification(){
  if(realtimeRunning) return;
  if(!selectedModel){showToast('未找到模型，请先刷新模型列表');return;}
  const item=getSelectedModelItem();
  if(item && item.has_meta===false){showToast('该模型缺少同名 yaml，无法验证');return;}
  realtimeRunning=true;
  await startCamera();
  setRealtimeButtonState(true);
  setResultRunning('实时检测中...');
  showToast('实时检测已开始');
  realtimeClassifyOnce();
  realtimeTimer=setInterval(realtimeClassifyOnce, REALTIME_INTERVAL_MS);
}

function stopRealtimeClassification(show=true){
  if(realtimeTimer){clearInterval(realtimeTimer);realtimeTimer=null;}
  const wasRunning=realtimeRunning;
  realtimeRunning=false;
  realtimeBusy=false;
  setRealtimeButtonState(false);
  updateCameraLifecycle();
  if(show && wasRunning) showToast('实时检测已停止');
}

function toggleRealtimeInferUI(){
  if(realtimeRunning) stopRealtimeClassification();
  else startRealtimeClassification();
}

async function initApp(){
  try{await loadCollectorState();await loadModels();await loadValidationImages();showToast('本地采集目录已就绪');}catch(err){showToast(err.message)}
  updateCameraLifecycle();
}
initApp();
