const toast = document.getElementById('toast');

function formatMessage(message){
  if(message === undefined || message === null) return '';
  if(typeof message === 'string') return message;
  if(message.message && typeof message.message === 'string') return message.message;
  if(Array.isArray(message)) return message.map(formatMessage).join('；');
  try{return JSON.stringify(message, null, 0);}catch(_){return String(message);}
}
function showToast(message){toast.textContent=formatMessage(message);toast.classList.add('show');clearTimeout(showToast.t);showToast.t=setTimeout(()=>toast.classList.remove('show'),2600)}

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


async function parseApiResponse(res){
  const text = await res.text();
  let data = {};
  if(text){
    try{data = JSON.parse(text);}catch(_){data = {message:text};}
  }
  if(!res.ok){
    const detail = data.detail || data.message || data.error || '请求失败';
    throw new Error(formatMessage(detail));
  }
  return data;
}
async function apiGet(url){const res=await fetch(url);return await parseApiResponse(res)}
async function apiPost(url, payload={}){const res=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});return await parseApiResponse(res)}


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
  stopRealtimeClassification(false);
  clearPendingClassificationCapture(false);
  modeToggle.textContent='切换工厂模式';
  modeToggle.classList.add('prod');
  factoryTabs.classList.add('hidden');
  factoryMode.classList.remove('active');
  productionMode.classList.add('active');
  startProductionMonitor();
}
function enterFactoryMode(){
  stopProductionMonitor();
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
  if(btn.dataset.page==='validatePage') refreshCppInferenceStatus(false);
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
let useCppCamera=false;
let backendCameraAvailable=false;
let cameraActive=false;
let cppCapturePreviewTimer=null;
let cppCapturePreviewStartedByCapturePage=false;
let cppCapturePreviewExternalStream=false;
let cppCapturePreviewIntervalMs=1000;
const CPP_PREVIEW_MIN_INTERVAL_MS=80;    // 最高约 12.5fps，避免 snapshot HTTP 请求过密
const CPP_PREVIEW_MAX_INTERVAL_MS=2000;  // 最低约 0.5fps
function normalizeCppPreviewFps(value){
  const fps=Number(value);
  if(!Number.isFinite(fps) || fps<=0) return 1;
  return Math.max(0.5, Math.min(fps, 12.5));
}
function setCppPreviewFps(value){
  const fps=normalizeCppPreviewFps(value);
  const interval=Math.round(1000/fps);
  cppCapturePreviewIntervalMs=Math.max(CPP_PREVIEW_MIN_INTERVAL_MS, Math.min(CPP_PREVIEW_MAX_INTERVAL_MS, interval));
  return cppCapturePreviewIntervalMs;
}
function getCppPreviewIntervalMs(){
  return cppCapturePreviewIntervalMs || 1000;
}
function getCppSnapshotFpsFromSettings(settings){
  if(!settings || typeof settings!=='object') return null;
  const value=settings.snapshot_fps ?? settings.cpp_snapshot_fps ?? settings.VISIONOPS_CPP_SNAPSHOT_FPS;
  const n=Number(value);
  return Number.isFinite(n) && n>0 ? n : null;
}
function restartCppPreviewLoopsAfterFpsChange(){
  // setInterval 的间隔在创建后不会自动变化，因此修改 Snapshot FPS 后要重建正在运行的前端预览循环。
  if(typeof startCppCapturePreviewImageLoop==='function' && cppCapturePreviewTimer && useCppCamera){
    startCppCapturePreviewImageLoop();
  }
  if(typeof startCppPreviewAutoLoop==='function' && cppPreviewTimer && cppPreviewModal && cppPreviewModal.classList.contains('active') && cppPreviewAuto){
    startCppPreviewAutoLoop();
  }
}
function applyCppPreviewFpsFromSettings(settings, restartLoops=false){
  const snapshotFps=getCppSnapshotFpsFromSettings(settings);
  if(snapshotFps===null) return getCppPreviewIntervalMs();
  const intervalMs=setCppPreviewFps(snapshotFps);
  const fpsText=(1000/intervalMs).toFixed(1);
  if(toggleCppPreviewAutoBtn){
    toggleCppPreviewAutoBtn.title=`当前前端预览刷新间隔 ${intervalMs}ms，约 ${fpsText}fps，跟随 C++ Snapshot FPS 设置`;
  }
  if(refreshCppPreviewFrameBtn){
    refreshCppPreviewFrameBtn.title=`手动刷新一帧；自动预览约 ${fpsText}fps`;
  }
  if(restartLoops) restartCppPreviewLoopsAfterFpsChange();
  return intervalMs;
}
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
let captureBusy=false;
let initialDefaultModeApplied=false;

function applyInitialDefaultMode(defaultMode){
  if(initialDefaultModeApplied) return;
  const mode=(defaultMode || '').toString().trim().toLowerCase();
  if(!mode) return;
  initialDefaultModeApplied=true;
  if(mode==='production' && !isProduction){
    enterProductionMode();
  }else if(mode==='factory' && isProduction){
    enterFactoryMode();
  }
}

function updateVisionBoxEffectiveStatus(status){
  const el=document.getElementById('visionBoxEffectiveStatus');
  if(!el || !status) return;
  const vb=status.vision_box || {};
  const disk=status.disk || {};
  const diskText=Number.isFinite(Number(disk.used_percent)) ? `${Number(disk.used_percent).toFixed(1)}%` : '--';
  const warn=!!disk.warning;
  el.classList.toggle('warn', warn);
  el.textContent=`已实现：设备ID=${vb.device_id || '--'}，客户ID=${vb.customer_id || '--'}，默认模式=${vb.default_mode || '--'}，模型目录=${vb.models_dir || '--'}，采集数据目录=${vb.data_dir || disk.path || '--'}，磁盘使用=${diskText}/${disk.warn_percent || '--'}%。${warn ? ' 已超过告警阈值' : ''}`;
}

async function refreshVisionBoxEffectiveStatus(){
  try{
    const data=await apiGet('/api/settings/vision-box/effective');
    updateVisionBoxEffectiveStatus(data);
    return data;
  }catch(_){
    return null;
  }
}

function formatOffsetSeconds(value){
  const n=Number(value);
  if(!Number.isFinite(n)) return '--';
  const abs=Math.abs(n);
  if(abs < 0.001) return `${(n*1000000).toFixed(1)} us`;
  if(abs < 1) return `${(n*1000).toFixed(2)} ms`;
  return `${n.toFixed(3)} s`;
}

function updateTimeSyncStatus(data){
  const el=document.getElementById('timeSyncStatus');
  if(!el) return;
  if(!data){
    el.textContent='时间同步状态：未读取';
    el.classList.remove('warn');
    return;
  }
  const st=data.status || {};
  const cfg=data.config || {};
  const sources=data.sources || [];
  const selected=st.selected_source || '--';
  const configured=cfg.ntp_server || '--';
  const synced=!!st.synced;
  const leap=st.leap_status || '--';
  const offset=formatOffsetSeconds(st.system_time_offset_sec);
  const lastOffset=formatOffsetSeconds(st.last_offset_sec);
  const sourceCount=Array.isArray(sources) ? sources.length : 0;
  el.classList.toggle('warn', !synced);
  el.textContent=`时间同步：${synced ? '已同步' : '未同步/未选中'}；配置源=${configured}；当前源=${selected}；Leap=${leap}；Stratum=${st.stratum ?? '--'}；系统偏差=${offset}；最近偏差=${lastOffset}；sources=${sourceCount}`;
}

async function refreshTimeSyncStatus(){
  try{
    const data=await apiGet('/api/settings/time-sync/status');
    updateTimeSyncStatus(data);
    return data;
  }catch(err){
    updateTimeSyncStatus(null);
    showToast(err.message || '读取时间同步状态失败');
    return null;
  }
}

async function testTimeSync(){
  try{
    const settings=collectSettingsFromForm();
    const saved=await apiPost('/api/settings',settings);
    runtimeSettingsCache=saved.settings || settings;
    fillSettingsForm(runtimeSettingsCache);
    const data=await apiPost('/api/settings/time-sync/test',{});
    updateTimeSyncStatus(data);
    showToast(data.message || '时间同步测试完成');
  }catch(err){
    showToast(err.message || '时间同步测试失败');
  }
}

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
  const visionBox=data.vision_box || {};
  const customerId=visionBox.customer_id || '';
  document.getElementById('uploadDeviceId').value=deviceId;
  if(customerId && document.getElementById('uploadCustomerId')) document.getElementById('uploadCustomerId').value=customerId;
  if(data.disk) updateVisionBoxEffectiveStatus({vision_box:visionBox,disk:data.disk});
  applyInitialDefaultMode(visionBox.default_mode);
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
  const visionBox=data.vision_box || {};
  const customerId=visionBox.customer_id || '';
  document.getElementById('uploadDeviceId').value=deviceId;
  if(customerId && document.getElementById('uploadCustomerId')) document.getElementById('uploadCustomerId').value=customerId;
  if(data.disk) updateVisionBoxEffectiveStatus({vision_box:visionBox,disk:data.disk});
  applyInitialDefaultMode(visionBox.default_mode);
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
  return isCapturePageVisible() || realtimeRunning || productionRunning;
}

function updateCameraLifecycle(){
  if(shouldKeepCameraRunning()) startCamera();
  else stopCamera();
}

async function getCppStreamStatusSafe(){
  try{return await apiGet('/api/cpp/stream/status');}
  catch(_){return null;}
}
function setBackendImageVisible(title,text){
  if(backendCamera){
    backendCamera.style.display='';
    backendCamera.classList.add('active','backend-active');
  }
  if(cameraVideo) cameraVideo.classList.remove('active');
  if(simulatedCamera) simulatedCamera.classList.remove('active');
  if(cameraStatusTitle) cameraStatusTitle.textContent=title || '摄像头画面';
  if(cameraStatusText) cameraStatusText.textContent=text || '';
}
function stopCppCapturePreviewImageLoop(){
  if(cppCapturePreviewTimer){
    clearInterval(cppCapturePreviewTimer);
    cppCapturePreviewTimer=null;
  }
}
function startCppCapturePreviewImageLoop(){
  stopCppCapturePreviewImageLoop();
  const refresh=()=>{
    if(!useCppCamera || !backendCamera) return;
    backendCamera.src='/api/cpp/stream/snapshot.jpg?t=' + Date.now();
  };
  refresh();
  cppCapturePreviewTimer=setInterval(refresh, getCppPreviewIntervalMs());
}
async function startCppCameraPreview(){
  try{
    const status=await getCppStreamStatusSafe();
    const alreadyRunning=!!(status && status.running);
    cppCapturePreviewExternalStream=alreadyRunning;
    cppCapturePreviewStartedByCapturePage=false;

    if(!alreadyRunning){
      await apiPost('/api/cpp/stream/start?mode=preview',{});
      cppCapturePreviewStartedByCapturePage=true;
      await sleepMs(500);
    }

    useCppCamera=true;
    useBackendCamera=false;
    useRealCamera=false;
    setBackendImageVisible(
      alreadyRunning ? 'C++ 当前相机流画面' : 'C++ 低延迟预览画面',
      alreadyRunning
        ? 'C++ 服务已有相机流在运行，拍照采集页复用其最新帧；离开本页不会强制停止外部启动的流'
        : 'C++ 服务正在以 preview 模式取流；离开拍照采集页后会自动停止以降低资源占用'
    );
    startCppCapturePreviewImageLoop();
    return true;
  }catch(err){
    console.warn('C++ preview start failed, fallback to Python camera_service',err);
    useCppCamera=false;
    cppCapturePreviewStartedByCapturePage=false;
    cppCapturePreviewExternalStream=false;
    stopCppCapturePreviewImageLoop();
    return false;
  }
}
async function stopCppCameraPreview(callBackend=true){
  stopCppCapturePreviewImageLoop();
  const shouldStopCpp=callBackend && useCppCamera && cppCapturePreviewStartedByCapturePage && !realtimeRunning && !productionRunning;
  useCppCamera=false;
  cppCapturePreviewExternalStream=false;
  const wasStartedByCapturePage=cppCapturePreviewStartedByCapturePage;
  cppCapturePreviewStartedByCapturePage=false;
  if(shouldStopCpp || (callBackend && wasStartedByCapturePage && !realtimeRunning && !productionRunning)){
    try{await apiPost('/api/cpp/stream/stop',{});}catch(err){console.warn('stop C++ preview failed',err);}
  }
}
async function captureCppSnapshotDataUrl(maxAttempts=5){
  let lastError=null;
  for(let i=0;i<maxAttempts;i++){
    try{
      const res=await fetch('/api/cpp/stream/snapshot.jpg?t='+Date.now(),{cache:'no-store'});
      if(res.ok){
        const blob=await res.blob();
        return await blobToDataUrl(blob);
      }
      lastError=new Error(`读取 C++ 当前帧失败：HTTP ${res.status}`);
    }catch(err){
      lastError=err;
    }
    await sleepMs(250);
  }
  throw lastError || new Error('读取 C++ 当前帧失败');
}

async function startCamera(){
  if(cameraActive) return;
  cameraActive=true;
  // v0.5.6：拍照采集页优先使用 C++ preview 模式作为唯一 RTSP 拉流 owner；失败时回退旧 Python MJPEG。
  if((backendCameraAvailable || useBackendCamera) && isCapturePageVisible()){
    const cppOk=await startCppCameraPreview();
    if(cppOk) return;
  }
  // 部署到 RK3588 + 海康 RTSP 时，旧路径使用后端 Python MJPEG 流；保留为 fallback，避免影响原有功能。
  if(backendCameraAvailable || useBackendCamera){
    useBackendCamera=true;
    useCppCamera=false;
    backendCamera.style.display='';
    backendCamera.src='/api/camera/stream?t=' + Date.now();
    backendCamera.classList.add('active','backend-active');
    cameraVideo.classList.remove('active');
    simulatedCamera.classList.remove('active');
    cameraStatusTitle.textContent='RTSP/后端摄像头画面';
    cameraStatusText.textContent='C++ 预览不可用，已回退到旧 Python 后端摄像头线程；离开拍照采集页后会自动停止以降低资源占用';
    return;
  }
  if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){cameraStatusTitle.textContent='模拟摄像头画面';cameraStatusText.textContent='当前浏览器不支持摄像头接口，点击按钮后保存模拟图片';return;}
  try{
    const stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});
    cameraVideo.srcObject=stream;useRealCamera=true;useBackendCamera=false;useCppCamera=false;simulatedCamera.classList.remove('active');backendCamera.classList.remove('active','backend-active');cameraVideo.classList.add('active');cameraStatusTitle.textContent='笔记本摄像头画面';cameraStatusText.textContent='点击上方按钮后，将当前帧直接保存到对应文件夹；离开本页后会自动关闭摄像头';
  }catch(err){useRealCamera=false;useBackendCamera=false;useCppCamera=false;simulatedCamera.classList.add('active');backendCamera.classList.remove('active','backend-active');cameraStatusTitle.textContent='模拟摄像头画面';cameraStatusText.textContent='未获得摄像头权限，点击按钮后保存模拟图片';}
}

async function stopCamera(callBackend=true){
  if(cameraVideo && cameraVideo.srcObject){
    cameraVideo.srcObject.getTracks().forEach(track=>track.stop());
    cameraVideo.srcObject=null;
  }
  useRealCamera=false;
  await stopCppCameraPreview(callBackend);
  if(backendCamera){
    backendCamera.removeAttribute('src');
    backendCamera.classList.remove('active','backend-active');
    backendCamera.style.display='none';
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
  if((useBackendCamera || useCppCamera) && backendCamera.complete && backendCamera.naturalWidth>0){
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
    if((useBackendCamera || useCppCamera) && backendCamera) backendCamera.classList.add('active','backend-active');
    else if(useRealCamera && cameraVideo) cameraVideo.classList.add('active');
    else if(simulatedCamera) simulatedCamera.classList.add('active');
  }
}
function setCaptureBusy(busy){
  captureBusy=!!busy;
  const btn=document.getElementById('captureAllBtn');
  if(btn){
    btn.disabled=captureBusy;
    btn.classList.toggle('loading', captureBusy);
    if(!btn.dataset.originalText) btn.dataset.originalText=btn.textContent || '取图';
    btn.textContent=captureBusy ? '取图中...' : btn.dataset.originalText;
  }
}

function makePreviewCard(item, folder){
  const card=document.createElement('div');
  card.className='preview-card';
  card.dataset.filename=item.filename || item.name || item.id || '';
  card.innerHTML=`<div class="preview-thumb-wrap"><img class="preview-img" src="${item.url}?t=${Date.now()}" alt="${item.filename}"><button class="delete-chip" title="删除图片">×</button></div><div class="preview-meta"><b title="${item.filename}">${item.filename}</b><span>${folderLabel(folder)} · ${item.mtime}</span></div>`;
  card.querySelector('.preview-img').addEventListener('click',()=>openImageViewer(item));
  card.querySelector('.delete-chip').addEventListener('click',(e)=>{e.stopPropagation();deletePreviewImage(item)});
  return card;
}

function prependCapturedItem(folder, item){
  if(!item) return false;
  if(!dataStore[folder]) dataStore[folder]=[];
  dataStore[folder]=[item, ...dataStore[folder].filter((x)=>x.filename!==item.filename && x.id!==item.id)].slice(0,120);

  if(currentFolder!==folder || !previewGrid) return true;

  const empty=previewGrid.querySelector('.empty-preview');
  if(empty) previewGrid.innerHTML='';

  const card=makePreviewCard(item, folder);
  previewGrid.prepend(card);

  while(previewGrid.children.length>120){
    previewGrid.removeChild(previewGrid.lastElementChild);
  }
  return true;
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
  if(useCppCamera){
    return await captureCppSnapshotDataUrl();
  }
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
  const payload={dataset:currentDataset,folder,device_id:deviceId,user_id:userId};
  // v0.5.6：C++ preview 模式下从 C++ snapshot 取当前帧并沿用原 /api/capture 保存逻辑；
  // 旧 Python 后端摄像头模式仍保持不传 image_data，由后端 camera_service 兜底保存 latest_frame。
  if(useCppCamera){
    payload.image_data=await captureCppSnapshotDataUrl();
  }else if(!useBackendCamera){
    payload.image_data=captureFrameAsJpeg();
  }
  const data=await apiPost('/api/capture',payload);
  syncCounts(data.counts);

  // 取图后不要每次 refreshFolder() 重载整个列表。
  // 否则会触发 /api/dataset/images + 大量缩略图 GET，图多时会明显卡顿。
  if(data.item){
    prependCapturedItem(folder, data.item);
  }else{
    await refreshFolder(currentFolder);
  }
  showToast(data.message);
}
async function handleCaptureButton(){
  if(captureBusy){
    showToast('正在取图，请稍候');
    return;
  }
  setCaptureBusy(true);
  try{
    const mode=(captureTaskMode && captureTaskMode.value) || 'classification';
    clearPendingClassificationCapture(false);
    if(mode==='detection'){
      await captureToFolder('all');
      return;
    }
    const imageData=await captureCurrentFrameDataUrl();
    showPendingClassificationCapture(imageData);
    showToast('图片已暂存，请选择合格或不合格');
  }catch(err){
    showToast(err.message || '取图失败');
  }finally{
    setCaptureBusy(false);
  }
}
async function savePendingClassificationCapture(label){
  if(!pendingClassificationImageData){showToast('没有暂存图片，请先取图');return;}
  try{
    const data=await apiPost('/api/capture/labeled',{dataset:currentDataset,image_data:pendingClassificationImageData,label,device_id:deviceId,user_id:userId});
    syncCounts(data.counts);
    clearPendingClassificationCapture(true);
    if(data.item){
      prependCapturedItem(label, data.item);
      if(currentFolder!==label) currentFolder=label;
      folderCards.forEach(b=>b.classList.toggle('active',b.dataset.folderCard===label));
      renderFolder(label, dataStore[label] || [data.item]);
    }else{
      await refreshFolder(label);
    }
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
    previewGrid.appendChild(makePreviewCard(item, folder));
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
  if(['seg','segment','segmentation','instance_segmentation','yolo_seg','yolov8_seg','mask_segmentation'].includes(t)) return 'segmentation';
  if(['detect','detection','yolo_detection','object_detection'].includes(t)) return 'detection';
  if(['roi_classification','roi_cls','two_stage_classification','pipeline_roi_classification'].includes(t)) return 'roi_classification';
  if(['cls','classify','classification','image_classification'].includes(t)) return 'classification';
  return t;
}
function isDetectionLikeTask(task){
  const t=normalizeTaskName(task);
  return t==='detection' || t==='obb_detection' || t==='segmentation' || t==='roi_classification';
}
function taskDisplayName(task){
  const t=normalizeTaskName(task);
  if(t==='classification') return '分类';
  if(t==='detection') return '检测';
  if(t==='obb_detection') return '旋转框检测';
  if(t==='segmentation') return '实例分割';
  if(t==='roi_classification') return 'ROI分类双模型';
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
let realtimeIntervalMs=1000; // v2.2：由系统设置动态控制
let productionDetectIntervalMs=1000;
let productionPollIntervalMs=1000;
let realtimeRunning=false;
let realtimeTimer=null;
let realtimeBusy=false;

const cppStatusCard=document.getElementById('cppInferenceStatusCard');
const cppServiceBadge=document.getElementById('cppServiceBadge');
const refreshCppStatusBtn=document.getElementById('refreshCppStatus');
const startCppPreviewStreamBtn=document.getElementById('startCppPreviewStream');
const startCppStreamBtn=document.getElementById('startCppStream');
const stopCppStreamBtn=document.getElementById('stopCppStream');
const openCppPreviewModalBtn=document.getElementById('openCppPreviewModal');
const cppPreviewModal=document.getElementById('cppPreviewModal');
const closeCppPreviewModalBtn=document.getElementById('closeCppPreviewModal');
const closeCppPreviewModalBottomBtn=document.getElementById('closeCppPreviewModalBottom');
const refreshCppPreviewFrameBtn=document.getElementById('refreshCppPreviewFrame');
const toggleCppPreviewAutoBtn=document.getElementById('toggleCppPreviewAuto');
const cppPreviewCanvas=document.getElementById('cppPreviewCanvas');
const cppPreviewEmpty=document.getElementById('cppPreviewEmpty');
const cppPreviewStatus=document.getElementById('cppPreviewStatus');
const cppPreviewCount=document.getElementById('cppPreviewCount');
const cppPreviewImageSize=document.getElementById('cppPreviewImageSize');
const cppPreviewUpdatedAt=document.getElementById('cppPreviewUpdatedAt');
const cppPreviewPredictions=document.getElementById('cppPreviewPredictions');
const cppRunningText=document.getElementById('cppRunningText');
const cppTaskText=document.getElementById('cppTaskText');
const cppModelText=document.getElementById('cppModelText');
const cppBackendText=document.getElementById('cppBackendText');
const cppCameraText=document.getElementById('cppCameraText');
const cppFpsText=document.getElementById('cppFpsText');
const cppLatencyText=document.getElementById('cppLatencyText');
const cppCaptureText=document.getElementById('cppCaptureText');
const cppPreprocessText=document.getElementById('cppPreprocessText');
const cppRknnText=document.getElementById('cppRknnText');
const cppPostprocessText=document.getElementById('cppPostprocessText');
const cppLastErrorText=document.getElementById('cppLastErrorText');
let cppStatusTimer=null;
let cppStatusBusy=false;
let cppActionBusy=false;
let cppPreviewTimer=null;
let cppPreviewBusy=false;
let cppPreviewAuto=true;
let cppPreviewLastObjectUrl='';

// 生产模式实时监控：复用模型验证实时检测接口，但独立渲染到生产模式大画面。
const productionPreview=document.getElementById('productionPreview');
const productionModelName=document.getElementById('productionModelName');
const productionStatus=document.getElementById('productionStatus');
const productionResultSummary=document.getElementById('productionResultSummary');
const productionLatency=document.getElementById('productionLatency');
const productionUpdatedAt=document.getElementById('productionUpdatedAt');
let productionRunning=false;
let productionTimer=null;
let productionBusy=false;
let productionLastPredictions=[];

function sleepMs(ms){return new Promise(resolve=>setTimeout(resolve, ms));}
function formatMetricNumber(value, digits=1){
  const n=Number(value);
  if(!Number.isFinite(n)) return '--';
  return n.toFixed(digits);
}
function formatMetricMs(value){
  const n=Number(value);
  if(!Number.isFinite(n)) return '--';
  return `${n.toFixed(1)} ms`;
}
function shortPathName(path){
  const s=String(path || '').trim();
  if(!s) return '--';
  const parts=s.split('/').filter(Boolean);
  return parts.length ? parts[parts.length-1] : s;
}
function setCppStatusBadge(state, text){
  if(!cppServiceBadge) return;
  cppServiceBadge.className=`cpp-service-badge ${state || 'unknown'}`;
  cppServiceBadge.textContent=text || '--';
}
function setCppStreamActionBusy(busy){
  cppActionBusy=!!busy;
  [startCppPreviewStreamBtn, startCppStreamBtn, stopCppStreamBtn, refreshCppStatusBtn, openCppPreviewModalBtn].forEach(btn=>{
    if(btn) btn.disabled=cppActionBusy;
  });
  if(startCppPreviewStreamBtn) startCppPreviewStreamBtn.textContent=busy ? '处理中...' : '启动 C++ 预览';
  if(startCppStreamBtn) startCppStreamBtn.textContent=busy ? '处理中...' : '启动 C++ 检测';
}
function updateCppStreamButtons(running, inferenceEnabled=false){
  const isRunning=!!running;
  const isDetect=!!inferenceEnabled;
  const isPreview=isRunning && !isDetect;
  if(startCppPreviewStreamBtn){
    startCppPreviewStreamBtn.disabled=cppActionBusy || isPreview;
    startCppPreviewStreamBtn.classList.toggle('active', isPreview);
  }
  if(startCppStreamBtn){
    startCppStreamBtn.disabled=cppActionBusy || isDetect;
    startCppStreamBtn.classList.toggle('active', isDetect);
  }
  if(stopCppStreamBtn){
    stopCppStreamBtn.disabled=cppActionBusy || !isRunning;
  }
  if(openCppPreviewModalBtn){
    openCppPreviewModalBtn.disabled=cppActionBusy;
  }
}
function getCppCameraSummary(health={}, status={}){
  const cameraType=status.camera_type || health.camera_type || '';
  const source=status.camera_source || health.camera_source || '';
  const width=Number(status.camera_width || health.camera_width || 0);
  const height=Number(status.camera_height || health.camera_height || 0);
  const fps=Number(status.camera_fps_target || status.camera_fps || health.camera_fps || 0);
  const fourcc=status.camera_fourcc || health.camera_fourcc || '';
  const sourceShort=source ? shortPathName(source) : '';
  const parts=[];
  if(cameraType && cameraType!=='auto') parts.push(cameraType);
  if(sourceShort) parts.push(sourceShort);
  if(width>0 && height>0) parts.push(`${width}×${height}`);
  if(fourcc) parts.push(fourcc);
  if(fps>0) parts.push(`${formatMetricNumber(fps,0)}fps`);
  return parts.join(' · ') || '--';
}
function isValidatePageActive(){
  const validatePage=document.getElementById('validatePage');
  return !!(validatePage && validatePage.classList.contains('active') && factoryMode && factoryMode.classList.contains('active'));
}
function renderCppStatusUnavailable(message){
  if(cppStatusCard) cppStatusCard.classList.add('error');
  setCppStatusBadge('error','连接失败');
  if(cppRunningText) cppRunningText.textContent='不可用';
  if(cppTaskText) cppTaskText.textContent='--';
  if(cppModelText) cppModelText.textContent='--';
  if(cppBackendText) cppBackendText.textContent='--';
  if(cppCameraText) cppCameraText.textContent='--';
  if(cppFpsText) cppFpsText.textContent='--';
  if(cppLatencyText) cppLatencyText.textContent='--';
  if(cppCaptureText) cppCaptureText.textContent='--';
  if(cppPreprocessText) cppPreprocessText.textContent='--';
  if(cppRknnText) cppRknnText.textContent='--';
  if(cppPostprocessText) cppPostprocessText.textContent='--';
  if(cppLastErrorText) cppLastErrorText.textContent=message || '无法连接 C++ 推理服务';
  updateCppStreamButtons(false,false);
}
function renderCppStatus({health={}, status={}, latest={}}={}){
  if(!cppStatusCard) return;
  cppStatusCard.classList.remove('collapsed','error');
  const healthOk=health && health.status==='ok';
  const running=!!status.running;
  const inferenceEnabled=!!status.inference_enabled;
  setCppStatusBadge(healthOk ? 'ok' : 'warning', healthOk ? '服务正常' : '状态异常');
  if(cppRunningText) cppRunningText.textContent=running ? (inferenceEnabled ? '检测中' : '预览中') : '未取流';
  updateCppStreamButtons(running, inferenceEnabled);
  const task=health.task || latest.task || '--';
  const input=Array.isArray(health.input_size) ? `${health.input_size[0]}×${health.input_size[1]}` : '';
  const classes=health.num_classes ? `${health.num_classes}类` : '';
  if(cppTaskText) cppTaskText.textContent=[taskDisplayName(task), input, classes].filter(Boolean).join(' · ') || '--';
  if(cppModelText){
    const model=health.model || health.model_path || latest.model || '--';
    cppModelText.textContent=shortPathName(model);
    cppModelText.title=model;
  }
  const backend=status.preprocess_backend_active || health.preprocess_backend_active || latest.preprocess_backend_active || latest.timing?.preprocess_backend || '--';
  const rgaMode=status.rga_mode_active || health.rga_mode_active || latest.rga_mode_active || '';
  if(cppBackendText) cppBackendText.textContent=[backend, rgaMode].filter(Boolean).join(' / ');
  if(cppCameraText){
    const cameraSummary=getCppCameraSummary(health,status);
    cppCameraText.textContent=cameraSummary;
    cppCameraText.title=cameraSummary;
  }
  if(cppFpsText){
    const cameraFps=formatMetricNumber(status.camera_fps,1);
    const detectFps=formatMetricNumber(status.detect_fps,1);
    cppFpsText.textContent=`${cameraFps} / ${detectFps}`;
  }
  const timing=latest.timing || {};
  const streamDetail=timing.stream_detail || {};
  const diagnostics=status.diagnostics || {};
  if(cppLatencyText) cppLatencyText.textContent=formatMetricMs(latest.latency_ms ?? status.latest_latency_ms);
  if(cppCaptureText) cppCaptureText.textContent=formatMetricMs(streamDetail.capture_read_ms ?? diagnostics.last_capture_read_ms);
  if(cppPreprocessText) cppPreprocessText.textContent=formatMetricMs(timing.preprocess_ms);
  if(cppRknnText) cppRknnText.textContent=formatMetricMs(timing.rknn && timing.rknn.total_ms);
  if(cppPostprocessText) cppPostprocessText.textContent=formatMetricMs(timing.postprocess_ms);
  const rawMessage=String(latest.message || '').trim();
  let statusInfo=status.last_error || '';
  if(!statusInfo){
    if(!running) statusInfo='实时流未启动；点击“启动 C++ 预览”只看画面，或点击“启动 C++ 检测”开始推理';
    else if(rawMessage && /not produced result|starting|no result/i.test(rawMessage)) statusInfo='取流已启动，正在等待首帧推理结果';
    else if(rawMessage) statusInfo=rawMessage;
    else statusInfo='无';
  }
  if(cppLastErrorText) cppLastErrorText.textContent=statusInfo;
}
async function refreshCppInferenceStatus(showMessage=false){
  if(!cppStatusCard || cppStatusBusy) return;
  cppStatusBusy=true;
  try{
    const [healthRes,statusRes,latestRes]=await Promise.allSettled([
      apiGet('/api/cpp/health'),
      apiGet('/api/cpp/stream/status'),
      apiGet('/api/cpp/stream/latest_result')
    ]);
    const health=healthRes.status==='fulfilled' ? healthRes.value : {};
    const status=statusRes.status==='fulfilled' ? statusRes.value : {};
    const latest=latestRes.status==='fulfilled' ? latestRes.value : {};
    if(healthRes.status==='rejected' && statusRes.status==='rejected'){
      throw new Error(healthRes.reason?.message || statusRes.reason?.message || 'C++ 服务不可用');
    }
    renderCppStatus({health,status,latest});
    if(showMessage) showToast('C++ 状态已刷新');
  }catch(err){
    renderCppStatusUnavailable(err.message || String(err));
    if(showMessage) showToast(err.message || 'C++ 状态刷新失败');
  }finally{
    cppStatusBusy=false;
  }
}
async function startCppStreamMode(mode='detect', showOkToast=true){
  if(!cppStatusCard || cppActionBusy) return false;
  const normalized=mode==='preview' ? 'preview' : 'detect';
  setCppStreamActionBusy(true);
  try{
    await apiPost(`/api/cpp/stream/start?mode=${encodeURIComponent(normalized)}`,{});
    if(showOkToast) showToast(normalized==='preview' ? 'C++ 预览已启动' : 'C++ 实时检测已启动');
    await sleepMs(normalized==='preview' ? 500 : 900);
    await refreshCppInferenceStatus(false);
    return true;
  }catch(err){
    renderCppStatusUnavailable(err.message || String(err));
    if(showOkToast) showToast(err.message || (normalized==='preview' ? '启动 C++ 预览失败' : '启动 C++ 检测失败'));
    return false;
  }finally{
    setCppStreamActionBusy(false);
  }
}
async function startCppPreviewStream(){
  return await startCppStreamMode('preview', true);
}
async function startCppStream(){
  return await startCppStreamMode('detect', true);
}
async function ensureCppStreamForPreview(){
  try{
    const status=await apiGet('/api/cpp/stream/status');
    if(status && status.running) return true;
  }catch(_){/* continue and try to start preview */}
  return await startCppStreamMode('preview', false);
}
async function stopCppStream(){
  if(!cppStatusCard || cppActionBusy) return;
  setCppStreamActionBusy(true);
  try{
    await apiPost('/api/cpp/stream/stop');
    showToast('C++ 流已停止');
    await sleepMs(300);
    await refreshCppInferenceStatus(false);
  }catch(err){
    renderCppStatusUnavailable(err.message || String(err));
    showToast(err.message || '停止 C++ 检测失败');
  }finally{
    setCppStreamActionBusy(false);
  }
}
function startCppStatusPolling(){
  if(!cppStatusCard || cppStatusTimer) return;
  refreshCppInferenceStatus(false);
  cppStatusTimer=setInterval(()=>{
    if(isValidatePageActive()) refreshCppInferenceStatus(false);
  },3000);
}

function setCppPreviewStatus(message, isError=false){
  if(!cppPreviewStatus) return;
  cppPreviewStatus.textContent=message || '--';
  cppPreviewStatus.classList.toggle('error', !!isError);
}
function clearCppPreviewCanvas(message='暂无预览图像'){
  if(cppPreviewCanvas){
    const ctx=cppPreviewCanvas.getContext('2d');
    cppPreviewCanvas.width=960;
    cppPreviewCanvas.height=540;
    ctx.clearRect(0,0,cppPreviewCanvas.width,cppPreviewCanvas.height);
  }
  if(cppPreviewEmpty){
    cppPreviewEmpty.textContent=message;
    cppPreviewEmpty.classList.add('active');
  }
}
function renderCppPreviewSide(latest={}, imageInfo={}){
  const preds=Array.isArray(latest.predictions) ? latest.predictions : [];
  if(cppPreviewCount) cppPreviewCount.textContent=String(preds.length || latest.count || 0);
  if(cppPreviewImageSize){
    const w=latest.image_width || imageInfo.width;
    const h=latest.image_height || imageInfo.height;
    cppPreviewImageSize.textContent=(w && h) ? `${w}×${h}` : '--';
  }
  if(cppPreviewUpdatedAt) cppPreviewUpdatedAt.textContent=new Date().toLocaleTimeString();
  if(cppPreviewPredictions){
    if(!preds.length){
      cppPreviewPredictions.textContent='暂无检测目标';
    }else{
      cppPreviewPredictions.innerHTML=preds.slice(0,8).map((p,idx)=>{
        const name=p.class_name || p.class_id || 'target';
        const conf=Number(p.confidence);
        const confText=Number.isFinite(conf) ? `${(conf*100).toFixed(1)}%` : '--';
        return `<div class="cpp-preview-pred"><b>${idx+1}. ${escapeHtml(String(name))}</b><span>${confText}</span></div>`;
      }).join('');
    }
  }
}
function drawCppPreviewFrame(img, latest={}){
  if(!cppPreviewCanvas) return;
  const naturalW=img.naturalWidth || img.width;
  const naturalH=img.naturalHeight || img.height;
  if(!naturalW || !naturalH) return;
  const maxW=1280;
  const scale=Math.min(1, maxW / naturalW);
  const canvasW=Math.max(1, Math.round(naturalW * scale));
  const canvasH=Math.max(1, Math.round(naturalH * scale));
  cppPreviewCanvas.width=canvasW;
  cppPreviewCanvas.height=canvasH;
  const ctx=cppPreviewCanvas.getContext('2d');
  ctx.clearRect(0,0,canvasW,canvasH);
  ctx.drawImage(img,0,0,canvasW,canvasH);
  const srcW=Number(latest.image_width) || naturalW;
  const srcH=Number(latest.image_height) || naturalH;
  const sx=canvasW / Math.max(1,srcW);
  const sy=canvasH / Math.max(1,srcH);
  const preds=Array.isArray(latest.predictions) ? latest.predictions : [];
  ctx.lineWidth=Math.max(2, Math.round(canvasW / 480));
  ctx.font=`${Math.max(14, Math.round(canvasW/70))}px sans-serif`;
  preds.forEach((p,idx)=>{
    const box=Array.isArray(p.bbox) ? p.bbox.map(Number) : null;
    if(!box || box.length<4 || box.some(v=>!Number.isFinite(v))) return;
    const x1=box[0]*sx;
    const y1=box[1]*sy;
    const x2=box[2]*sx;
    const y2=box[3]*sy;
    const w=Math.max(1,x2-x1);
    const h=Math.max(1,y2-y1);
    ctx.strokeStyle='#22c55e';
    ctx.fillStyle='rgba(34,197,94,.14)';
    ctx.strokeRect(x1,y1,w,h);
    ctx.fillRect(x1,y1,w,h);
    const conf=Number(p.confidence);
    const label=`${p.class_name || p.class_id || 'target'} ${Number.isFinite(conf) ? (conf*100).toFixed(1)+'%' : ''}`.trim();
    const textW=ctx.measureText(label).width;
    const labelH=Math.max(22, Math.round(canvasW/55));
    const ly=Math.max(0,y1-labelH-2);
    ctx.fillStyle='rgba(15,23,42,.88)';
    ctx.fillRect(x1,ly,Math.min(textW+14,canvasW-x1),labelH);
    ctx.fillStyle='#fff';
    ctx.fillText(label,x1+7,ly+labelH-7);
    const center=getPredictionCenter(p, box);
    if(center){
      ctx.beginPath();
      ctx.arc(center[0]*sx, center[1]*sy, Math.max(3,ctx.lineWidth+1), 0, Math.PI*2);
      ctx.fillStyle='#ef4444';
      ctx.fill();
    }
  });
  if(cppPreviewEmpty) cppPreviewEmpty.classList.remove('active');
  renderCppPreviewSide(latest,{width:naturalW,height:naturalH});
}
async function loadCppSnapshotImage(){
  const resp=await fetch(`/api/cpp/stream/snapshot.jpg?t=${Date.now()}`, {cache:'no-store'});
  if(!resp.ok){
    let detail='快照请求失败';
    try{detail=(await resp.json()).message || (await resp.text()) || detail;}catch(_){/* ignore */}
    throw new Error(detail);
  }
  const contentType=resp.headers.get('content-type') || '';
  if(!contentType.includes('image')){
    let message='当前没有可用快照';
    try{const data=await resp.json(); message=data.message || data.detail || message;}catch(_){/* ignore */}
    throw new Error(message);
  }
  const blob=await resp.blob();
  const url=URL.createObjectURL(blob);
  return await new Promise((resolve,reject)=>{
    const img=new Image();
    img.onload=()=>resolve({img,url});
    img.onerror=()=>{URL.revokeObjectURL(url); reject(new Error('快照图像解码失败'));};
    img.src=url;
  });
}
async function refreshCppPreviewFrame(showToastOnSuccess=false){
  if(!cppPreviewModal || !cppPreviewModal.classList.contains('active') || cppPreviewBusy) return;
  cppPreviewBusy=true;
  if(refreshCppPreviewFrameBtn) refreshCppPreviewFrameBtn.disabled=true;
  try{
    setCppPreviewStatus(`正在刷新预览...（约 ${(1000/getCppPreviewIntervalMs()).toFixed(1)}fps）`);
    const [status, latest]=await Promise.all([
      apiGet('/api/cpp/stream/status').catch(()=>({})),
      apiGet('/api/cpp/stream/latest_result').catch(()=>({}))
    ]);
    if(!status.running){
      clearCppPreviewCanvas('实时流未启动；请先点击“启动 C++ 预览”或“启动 C++ 检测”。');
      renderCppPreviewSide(latest,{});
      setCppPreviewStatus('实时流未启动', true);
      return;
    }
    const {img,url}=await loadCppSnapshotImage();
    if(cppPreviewLastObjectUrl) URL.revokeObjectURL(cppPreviewLastObjectUrl);
    cppPreviewLastObjectUrl=url;
    drawCppPreviewFrame(img,latest || {});
    const count=Array.isArray(latest.predictions) ? latest.predictions.length : (latest.count || 0);
    setCppPreviewStatus(`已刷新：${count} 个目标，${new Date().toLocaleTimeString()}`);
    if(showToastOnSuccess) showToast('C++ 预览已刷新');
  }catch(err){
    clearCppPreviewCanvas(err.message || '预览刷新失败');
    setCppPreviewStatus(err.message || '预览刷新失败', true);
  }finally{
    cppPreviewBusy=false;
    if(refreshCppPreviewFrameBtn) refreshCppPreviewFrameBtn.disabled=false;
  }
}
function startCppPreviewAutoLoop(){
  if(cppPreviewTimer) clearInterval(cppPreviewTimer);
  const intervalMs=getCppPreviewIntervalMs();
  cppPreviewTimer=setInterval(()=>{
    if(cppPreviewModal && cppPreviewModal.classList.contains('active') && cppPreviewAuto){
      refreshCppPreviewFrame(false);
    }
  },intervalMs);
}
function stopCppPreviewAutoLoop(){
  if(cppPreviewTimer){
    clearInterval(cppPreviewTimer);
    cppPreviewTimer=null;
  }
}
function updateCppPreviewAutoButton(){
  if(!toggleCppPreviewAutoBtn) return;
  toggleCppPreviewAutoBtn.textContent=cppPreviewAuto ? '暂停自动预览' : '恢复自动预览';
  toggleCppPreviewAutoBtn.classList.toggle('active', cppPreviewAuto);
}
async function openCppPreviewModal(){
  if(!cppPreviewModal) return;
  cppPreviewModal.classList.add('active');
  cppPreviewAuto=true;
  updateCppPreviewAutoButton();
  setCppPreviewStatus('正在打开预览...');
  const ok=await ensureCppStreamForPreview();
  if(!ok){
    clearCppPreviewCanvas('C++ 预览启动失败，请检查 /api/cpp/health 和 C++ 服务日志。');
    setCppPreviewStatus('C++ 预览启动失败', true);
    startCppPreviewAutoLoop();
    return;
  }
  await refreshCppPreviewFrame(false);
  startCppPreviewAutoLoop();
}
function closeCppPreviewModal(){
  if(!cppPreviewModal) return;
  cppPreviewModal.classList.remove('active');
  stopCppPreviewAutoLoop();
  setCppPreviewStatus('预览已关闭');
}

function updateSelectedModelUI(){
  if(!selectedModelNameEl) return;
  const item=getSelectedModelItem ? getSelectedModelItem() : null;
  if(!selectedModel){selectedModelNameEl.textContent='未找到模型';return;}
  if(item && item.has_meta!==false){
    const taskText=item.task_label || item.task || '未知任务';
    const clsText=item.num_classes ? ` · ${item.num_classes}类` : '';
    const pipelineText=item.is_pipeline ? ' · 双模型推理' : '';
    selectedModelNameEl.textContent=`${selectedModel}（${taskText}${clsText}${pipelineText}）`;
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
    const pipelineText=item.is_pipeline ? '双模型推理' : '';
    const sub=[item.label || taskText, pipelineText, classText, customerText, sizeText].filter(Boolean).join(' · ');
    const disabled=item.has_meta===false ? ' data-missing-meta="1"' : '';
    const extraClass=item.is_pipeline ? ' pipeline-model-card' : '';
    return `<button class="model-card ${active}${extraClass}" data-model-name="${escapeHtml(item.name)}" title="${escapeHtml(item.pipeline_config || item.meta_path || item.path || item.name)}"${disabled}><b>${escapeHtml(item.name)}</b><span>${escapeHtml(sub)}</span></button>`;
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

function appendCacheBuster(url){
  const raw=String(url || '');
  if(!raw) return raw;
  const sep=raw.includes('?') ? '&' : '?';
  return `${raw}${sep}_=${Date.now()}`;
}

function ensurePreviewImage(container, imgId, canvasId, wrapClass='', alt='实时画面'){
  if(!container) return null;
  let img=document.getElementById(imgId);
  let canvas=document.getElementById(canvasId);
  const wrap=container.querySelector('.preview-canvas-wrap');
  if(!img || !canvas || !wrap){
    container.innerHTML=`
      <div class="preview-canvas-wrap ${wrapClass}">
        <img id="${imgId}" alt="${escapeHtml(alt)}">
        <canvas id="${canvasId}" aria-hidden="true"></canvas>
      </div>
    `;
    img=document.getElementById(imgId);
    canvas=document.getElementById(canvasId);
  }
  if(img) img.alt=alt;
  return {img,canvas};
}

function swapPreviewImageAfterLoad(img, url, onReady){
  if(!img || !url) return;

  // 每次图像更新都生成 token，避免旧请求 / 旧 onload 晚到后把新检测框清掉。
  const token=`${Date.now()}_${Math.random().toString(16).slice(2)}`;
  img.dataset.previewToken=token;

  const bindVisibleImageAndDraw=()=>{
    if(img.dataset.previewToken!==token) return;

    let called=false;
    const ready=()=>{
      if(called || img.dataset.previewToken!==token) return;
      called=true;
      requestAnimationFrame(()=>{
        if(img.dataset.previewToken===token && typeof onReady==='function') onReady();
      });
    };

    // 清理上一帧遗留的 onload/onerror。上一版黑闪修复里 onload 会残留，
    // 新 src 加载完成后可能再次触发旧回调，导致 canvas 检测框被旧 predictions 清空。
    img.onload=ready;
    img.onerror=()=>console.warn('preview image load failed:', url);
    img.src=url;

    // 对于已经预加载/浏览器缓存命中的图片，onload 可能很快或已经 complete，
    // 这里补一次 ready，确保检测框在可见 img 尺寸稳定后绘制。
    if(img.complete && img.naturalWidth>0){
      ready();
    }
  };

  // 首帧没有旧图时直接加载；之后先预加载新图，成功后再替换，避免画面黑闪。
  if(!img.getAttribute('src') || !img.naturalWidth){
    bindVisibleImageAndDraw();
    return;
  }

  const next=new Image();
  next.onload=bindVisibleImageAndDraw;
  next.onerror=()=>console.warn('preview image preload failed:', url);
  next.src=url;
}

function renderPreviewImage(url, alt='测试图片', detections=[]){
  if(!selectedImagePreview) return;
  const view=ensurePreviewImage(selectedImagePreview, 'resultPreviewImg', 'detectionOverlayCanvas', '', alt);
  if(!view || !view.img) return;
  const finalUrl=appendCacheBuster(url);
  swapPreviewImageAfterLoad(view.img, finalUrl, ()=>drawDetectionOverlay(detections));
}
function clearDetectionOverlay(){
  const canvas=document.getElementById('detectionOverlayCanvas');
  if(!canvas) return;
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  canvas.style.display='none';
}

function normalizePolygonPoints(points, canvasWidth, canvasHeight){
  if(!Array.isArray(points) || points.length<3) return null;
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
  return normalized.length>=3 ? normalized : null;
}
function getValidMaskSegments(pred, canvasWidth, canvasHeight){
  const mask=pred && pred.mask && typeof pred.mask==='object' ? pred.mask : null;
  if(!mask) return [];
  const segments=[];
  if(Array.isArray(mask.segments)){
    mask.segments.forEach((seg)=>{
      const normalized=normalizePolygonPoints(seg, canvasWidth, canvasHeight);
      if(normalized) segments.push(normalized);
    });
  }
  if(!segments.length && Array.isArray(mask.polygon)){
    const normalized=normalizePolygonPoints(mask.polygon, canvasWidth, canvasHeight);
    if(normalized) segments.push(normalized);
  }
  return segments;
}
function drawMaskSegmentsOverlay(ctx, segments){
  if(!Array.isArray(segments) || !segments.length) return;
  segments.forEach((points)=>{
    if(!points || points.length<3) return;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for(let i=1;i<points.length;i++) ctx.lineTo(points[i][0], points[i][1]);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  });
}
function getMaskLabelAnchor(segments, fallbackBox=null){
  if(Array.isArray(segments) && segments.length){
    let best=null;
    segments.forEach((seg)=>{
      (seg || []).forEach((p)=>{
        const x=Number(p && p[0]);
        const y=Number(p && p[1]);
        if(!Number.isFinite(x) || !Number.isFinite(y)) return;
        if(!best || y<best[1] || (y===best[1] && x<best[0])) best=[x,y];
      });
    });
    if(best) return best;
  }
  if(fallbackBox && fallbackBox.length>=2) return [fallbackBox[0], fallbackBox[1]];
  return [0,0];
}
function formatMaskArea(pred){
  const area=Number(pred && pred.mask && pred.mask.area);
  if(!Number.isFinite(area)) return '--';
  return area>=1000 ? `${Math.round(area)} px` : `${area.toFixed(1)} px`;
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
function getCurrentAlgorithmSettings(){
  return (runtimeSettingsCache && runtimeSettingsCache.algorithm) || {};
}
function shouldShowDetectionCenter(){
  const det=(getCurrentAlgorithmSettings().detection)||{};
  return det.show_center !== false;
}
function getMaskOverlayAlpha(){
  const seg=(getCurrentAlgorithmSettings().segmentation)||{};
  return clampNumber(seg.mask_alpha,0.24,0,1);
}

function getRoiObjectFromPrediction(pred){
  return pred && pred.roi && typeof pred.roi==='object' ? pred.roi : null;
}
function getRoiBoxFromPrediction(pred){
  const roi=getRoiObjectFromPrediction(pred);
  if(!roi || !Array.isArray(roi.bbox) || roi.bbox.length<4) return null;
  const box=roi.bbox.map(Number);
  return box.some((x)=>!Number.isFinite(x)) ? null : box;
}
function formatRoiMode(roi){
  if(!roi || typeof roi!=='object') return '--';
  const mode=roi.mode || '--';
  const pipeline=roi.pipeline_mode || '';
  return pipeline && pipeline!==mode ? `${pipeline} / ${mode}` : mode;
}
function formatRelativeBox(rel){
  if(!rel || typeof rel!=='object') return '--';
  const keys=['x1','y1','x2','y2'];
  return keys.map((k)=>{
    const v=Number(rel[k]);
    return `${k}=${Number.isFinite(v) ? v.toFixed(3) : '--'}`;
  }).join(', ');
}
function drawRoiBoxOverlay(ctx, roiBox, canvasWidth, canvasHeight){
  if(!roiBox || roiBox.length<4) return;
  const x1=Math.max(0, Math.min(canvasWidth, roiBox[0]));
  const y1=Math.max(0, Math.min(canvasHeight, roiBox[1]));
  const x2=Math.max(0, Math.min(canvasWidth, roiBox[2]));
  const y2=Math.max(0, Math.min(canvasHeight, roiBox[3]));
  const w=Math.max(0, x2-x1);
  const h=Math.max(0, y2-y1);
  if(w<2 || h<2) return;
  ctx.save();
  ctx.setLineDash([]);
  ctx.lineWidth=Math.max(3, Math.round(canvasWidth/360));
  ctx.strokeStyle='#f97316';
  ctx.fillStyle='rgba(249,115,22,.10)';
  ctx.fillRect(x1,y1,w,h);
  ctx.strokeRect(x1,y1,w,h);

  const label='Final ROI';
  const pad=Math.max(5, Math.round(canvasWidth/220));
  const oldFont=ctx.font;
  ctx.font=`${Math.max(14, Math.round(canvasWidth/58))}px sans-serif`;
  const textW=ctx.measureText(label).width;
  const labelH=Math.max(20, Math.round(canvasWidth/48));
  const lx=Math.max(0, Math.min(canvasWidth-(textW+pad*2), x1));
  const ly=Math.min(canvasHeight-labelH, y2+2);
  //ctx.fillStyle='rgba(249,115,22,.94)';
  //ctx.fillRect(lx,ly,textW+pad*2,labelH);
  //ctx.fillStyle='#fff';
  //ctx.fillText(label,lx+pad,ly+pad/2);
  //ctx.font=oldFont;
  ctx.restore();
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
    const maskSegments=getValidMaskSegments(pred, canvas.width, canvas.height);
    const obbPoints=getValidObbPoints(pred, canvas.width, canvas.height);
    const box=Array.isArray(pred.bbox) ? pred.bbox.map(Number) : null;

    let labelAnchor=null;
    let hasShape=false;
    if(maskSegments.length){
      ctx.strokeStyle='#f97316';
      ctx.fillStyle=`rgba(249,115,22,${getMaskOverlayAlpha()})`;
      ctx.lineWidth=Math.max(2, Math.round(canvas.width/420));
      drawMaskSegmentsOverlay(ctx, maskSegments);
      // segmentation 只绘制 mask polygon，不再叠加 bbox 矩形框。
      labelAnchor=getMaskLabelAnchor(maskSegments, box);
      hasShape=true;
    }else if(obbPoints){
      ctx.strokeStyle='#06b6d4';
      ctx.fillStyle='rgba(6,182,212,.16)';
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
        ctx.strokeStyle='#22c55e';
        ctx.fillStyle='rgba(34,197,94,.18)';
        ctx.fillRect(x1,y1,w,h);
        ctx.strokeRect(x1,y1,w,h);
        labelAnchor=[x1,y1];
        hasShape=true;
      }
    }
    if(!hasShape) return;

    // ROI 分类双模型：绿色框是 detector bbox，橙色实线框是最终送入分类模型的 final ROI。
    const roiBox=getRoiBoxFromPrediction(pred);
    if(roiBox) drawRoiBoxOverlay(ctx, roiBox, canvas.width, canvas.height);

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
    if(shouldShowDetectionCenter() && center && Number.isFinite(center[0]) && Number.isFinite(center[1])){
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
  if(productionLastPredictions.length) drawProductionOverlay(productionLastPredictions);
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
    const normalizedTask=normalizeTaskName(task);
    const taskText=taskDisplayName(normalizedTask);
    const updatedAt=data.updated_at || new Date().toLocaleTimeString();

    if(normalizedTask==='roi_classification'){
      const rr=data.result || {};
      const finalName=rr.class_name || data.final_label || data.final_decision || (predictions[0] && predictions[0].class_name) || '未识别';
      const finalConf=Number(rr.confidence ?? data.final_confidence ?? (predictions[0] && predictions[0].confidence));
      const finalConfText=Number.isFinite(finalConf) ? `${(finalConf*100).toFixed(1)}%` : (rr.confidence_percent || '--');
      resultClassName.textContent=finalName;
      resultConfidence.textContent=`${taskText} · ${finalConfText} · ${predictions.length} 个目标`;
      resultLatency.textContent=`耗时 ${data.latency_ms ?? '--'} ms · 更新 ${updatedAt}`;

      if(topkResult){
        if(predictions.length){
          topkResult.innerHTML='<b>ROI 分类明细</b>'+predictions.map((p,idx)=>{
            const bbox=Array.isArray(p.bbox) ? p.bbox : [];
            const center=getPredictionCenter(p);
            const roi=p.roi || (idx===0 ? data.roi : null) || {};
            const roiBox=getRoiBoxFromPrediction({...p, roi});
            const det=p.detector || {};
            const clsPred=p.classifier || {};
            const detConf=Number(det.confidence);
            const clsConf=Number(clsPred.confidence ?? p.confidence);
            const detText=`${escapeHtml(det.class_name || '目标')} ${Number.isFinite(detConf) ? (detConf*100).toFixed(1)+'%' : '--'}`;
            const clsText=`${escapeHtml(p.class_name || clsPred.class_name || '未识别')} ${Number.isFinite(clsConf) ? (clsConf*100).toFixed(1)+'%' : '--'}`;
            const roiMode=formatRoiMode(roi);
            const relText=formatRelativeBox(roi.relative_box);
            return `<div class="target-row target-row-roi">
              <span class="target-index">目标${idx+1}</span>
              <span class="target-class">分类：${clsText}</span>
              <span>检测：${detText}</span>
              <span>检测框：${escapeHtml(formatBBox(bbox))}</span>
              <span class="target-final-roi">Final ROI：${escapeHtml(formatBBox(roiBox))}</span>
              <span>ROI模式：${escapeHtml(roiMode)}</span>
              <span>ROI比例：${escapeHtml(relText)}</span>
              <span>中心点：${escapeHtml(formatPoint(center))}</span>
              <span>更新时间：${escapeHtml(updatedAt)}</span>
            </div>`;
          }).join('');
        }else{
          topkResult.innerHTML='<b>ROI 分类明细</b><div class="target-row empty">未检测到目标</div>';
        }
      }
      drawDetectionOverlay(predictions);
      if(!options.silent) showToast(`${taskText}完成：${finalName} ${finalConfText}`);
      return;
    }

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
          const maskText=normalizeTaskName(task)==='segmentation' ? `<span>面积：${escapeHtml(formatMaskArea(p))}</span>` : '';
          return `<div class="target-row target-row-inline ${normalizeTaskName(task)==='segmentation' ? 'target-row-seg' : ''}">
            <span class="target-index">目标${idx+1}</span>
            <span class="target-class">${cls}</span>
            <span>位置：${escapeHtml(formatBBox(bbox))}</span>
            <span>中心点：${escapeHtml(formatPoint(center))}</span>
            ${maskText}
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
if(startCppPreviewStreamBtn){startCppPreviewStreamBtn.addEventListener('click',startCppPreviewStream);}
if(startCppStreamBtn){startCppStreamBtn.addEventListener('click',startCppStream);}
if(stopCppStreamBtn){stopCppStreamBtn.addEventListener('click',stopCppStream);}
if(refreshCppStatusBtn){refreshCppStatusBtn.addEventListener('click',()=>refreshCppInferenceStatus(true));}
if(openCppPreviewModalBtn){openCppPreviewModalBtn.addEventListener('click',openCppPreviewModal);}
if(closeCppPreviewModalBtn){closeCppPreviewModalBtn.addEventListener('click',closeCppPreviewModal);}
if(closeCppPreviewModalBottomBtn){closeCppPreviewModalBottomBtn.addEventListener('click',closeCppPreviewModal);}
if(refreshCppPreviewFrameBtn){refreshCppPreviewFrameBtn.addEventListener('click',()=>refreshCppPreviewFrame(true));}
if(toggleCppPreviewAutoBtn){toggleCppPreviewAutoBtn.addEventListener('click',()=>{cppPreviewAuto=!cppPreviewAuto;updateCppPreviewAutoButton();if(cppPreviewAuto) refreshCppPreviewFrame(false);});}
if(cppPreviewModal){cppPreviewModal.addEventListener('click',(e)=>{if(e.target===cppPreviewModal) closeCppPreviewModal();});}


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
      const url=data.realtime.url || '/api/validation/realtime_image/realtime_latest.jpg';
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
  realtimeTimer=setInterval(realtimeClassifyOnce, realtimeIntervalMs);
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

function getProductionModelName(){
  if(selectedModel){
    const item=getSelectedModelItem();
    if(item && item.has_meta!==false) return selectedModel;
  }
  const usable=modelItems.find((item)=>item.has_meta!==false);
  return usable ? usable.name : '';
}

function clearProductionOverlay(){
  const canvas=document.getElementById('productionOverlayCanvas');
  if(!canvas) return;
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  canvas.style.display='none';
}

function renderProductionPreview(url, detections=[]){
  if(!productionPreview) return;
  productionLastPredictions=Array.isArray(detections) ? detections : [];
  const view=ensurePreviewImage(productionPreview, 'productionPreviewImg', 'productionOverlayCanvas', 'production-canvas-wrap', '生产模式实时画面');
  if(!view || !view.img) return;
  const finalUrl=appendCacheBuster(url);
  swapPreviewImageAfterLoad(view.img, finalUrl, ()=>drawProductionOverlay(productionLastPredictions));
}

function drawProductionOverlay(predictions=[]){
  const canvas=document.getElementById('productionOverlayCanvas');
  const img=document.getElementById('productionPreviewImg');
  const wrap=productionPreview ? productionPreview.querySelector('.preview-canvas-wrap') : null;
  if(!canvas || !img || !wrap) return;
  if(!Array.isArray(predictions) || predictions.length===0 || !img.naturalWidth || !img.naturalHeight){
    clearProductionOverlay();
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
    const rawBox=Array.isArray(pred.bbox) ? pred.bbox.map(Number) : null;
    const maskSegments=getValidMaskSegments(pred, canvas.width, canvas.height);
    const points=pred.obb && Array.isArray(pred.obb.points) ? pred.obb.points : null;
    let box=rawBox;
    if(points && points.length>=4){
      const xs=points.map(p=>Number(p[0])).filter(Number.isFinite);
      const ys=points.map(p=>Number(p[1])).filter(Number.isFinite);
      if(xs.length && ys.length) box=[Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    }
    if(!box || box.length<4 || box.some((x)=>!Number.isFinite(x))) return;
    const x1=Math.max(0, Math.min(canvas.width, box[0]));
    const y1=Math.max(0, Math.min(canvas.height, box[1]));
    const x2=Math.max(0, Math.min(canvas.width, box[2]));
    const y2=Math.max(0, Math.min(canvas.height, box[3]));
    const w=Math.max(0, x2-x1);
    const h=Math.max(0, y2-y1);
    if(w<2 || h<2) return;
    const cls=String(pred.class_name ?? pred.class ?? pred.label ?? pred.class_id ?? '目标');
    const conf=Number(pred.confidence ?? pred.score);
    const label=Number.isFinite(conf) ? `${cls} ${(conf*100).toFixed(1)}%` : cls;

    if(maskSegments.length){
      ctx.strokeStyle='#f97316';
      ctx.fillStyle=`rgba(249,115,22,${getMaskOverlayAlpha()})`;
      ctx.lineWidth=Math.max(2, Math.round(canvas.width/420));
      drawMaskSegmentsOverlay(ctx, maskSegments);
      // segmentation 只绘制 mask polygon，不再叠加 bbox 矩形框。
    }else if(points && points.length>=4){
      ctx.strokeStyle='#06b6d4';
      ctx.fillStyle='rgba(6,182,212,.16)';
      ctx.beginPath();
      points.forEach((pt,idx)=>{
        const px=Math.max(0, Math.min(canvas.width, Number(pt[0])));
        const py=Math.max(0, Math.min(canvas.height, Number(pt[1])));
        if(idx===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }else{
      ctx.strokeStyle='#22c55e';
      ctx.fillStyle='rgba(34,197,94,.18)';
      ctx.fillRect(x1,y1,w,h);
      ctx.strokeRect(x1,y1,w,h);
    }

    const roiBox=getRoiBoxFromPrediction(pred);
    if(roiBox) drawRoiBoxOverlay(ctx, roiBox, canvas.width, canvas.height);

    const pad=Math.max(6, Math.round(canvas.width/180));
    const textW=ctx.measureText(label).width;
    const labelH=Math.max(24, Math.round(canvas.width/36));
    const lx=Math.max(0, Math.min(canvas.width-(textW+pad*2), x1));
    const ly=Math.max(0, y1-labelH-2);
    ctx.fillStyle='rgba(15,23,42,.90)';
    ctx.fillRect(lx,ly,textW+pad*2,labelH);
    ctx.fillStyle='#fff';
    ctx.fillText(label,lx+pad,ly+pad/2);

    const center=getPredictionCenter(pred, box);
    if(shouldShowDetectionCenter() && center && Number.isFinite(center[0]) && Number.isFinite(center[1])){
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
    }
  });
}

async function refreshProductionPushStatus(){
  if(!productionRunning) return;

  try{
    const status=await apiGet('/api/production/push/status');
    const latest=status.latest_result || null;

    if(productionModelName) productionModelName.textContent=status.model_name || getProductionModelName() || '--';

    if(!status.running){
      if(productionStatus) productionStatus.textContent='已停止';
      return;
    }

    const pushed=status.latest_push_response && Number.isFinite(Number(status.latest_push_response.pushed_clients))
      ? Number(status.latest_push_response.pushed_clients)
      : null;

    if(productionStatus){
      productionStatus.textContent=pushed===null
        ? '连续检测中'
        : `连续检测中 / 已连接上位机 ${pushed}`;
    }

    if(status.latest_error){
      if(productionResultSummary) productionResultSummary.textContent=status.latest_error;
      return;
    }

    if(!latest){
      if(productionResultSummary) productionResultSummary.textContent='等待第一帧检测结果';
      return;
    }

    const task=normalizeTaskName(latest.task || '');
    const preds=Array.isArray(latest.predictions) ? latest.predictions : [];

    if(productionModelName) productionModelName.textContent=status.model_name || getProductionModelName() || '--';
    if(productionLatency) productionLatency.textContent=`${latest.latency_ms ?? '--'} ms`;
    if(productionUpdatedAt) productionUpdatedAt.textContent=new Date().toLocaleTimeString();

    if(latest.realtime){
      const url=latest.realtime.url || '/api/validation/realtime_image/realtime_latest.jpg';
      renderProductionPreview(url, preds);
    }

    if(isDetectionLikeTask(task)){
      const taskText=taskDisplayName(task);
      if(task==='roi_classification'){
        const r=latest.result || {};
        const firstPred=preds[0] || {};
        const roi=firstPred.roi || latest.roi || {};
        const roiMode=formatRoiMode(roi);
        if(productionResultSummary) productionResultSummary.textContent=`${taskText}：${r.class_name || '未识别'} ${r.confidence_percent || ''} · ROI ${roiMode}`;
      }else{
        if(productionResultSummary) productionResultSummary.textContent=`${taskText}到 ${preds.length} 个目标`;
      }
    }else{
      const r=latest.result || {};
      if(productionResultSummary) productionResultSummary.textContent=`${r.class_name || '未识别'} ${r.confidence_percent || ''}`;
    }
  }catch(err){
    if(productionStatus) productionStatus.textContent='状态读取异常';
    if(productionResultSummary) productionResultSummary.textContent=err.message || '读取生产检测状态失败';
  }
}

async function startProductionMonitor(){
  if(productionRunning) return;
  if(!modelItems.length){
    try{await loadModels();}catch(_e){}
  }
  const modelName=getProductionModelName();
  if(!modelName){
    if(productionStatus) productionStatus.textContent='未找到可用模型';
    if(productionResultSummary) productionResultSummary.textContent='请先部署模型';
    updateCameraLifecycle();
    return;
  }

  productionRunning=true;
  productionBusy=false;
  productionLastPredictions=[];
  if(productionModelName) productionModelName.textContent=modelName;
  if(productionStatus) productionStatus.textContent='启动连续检测中';
  if(productionResultSummary) productionResultSummary.textContent='等待检测';
  if(productionLatency) productionLatency.textContent='--';
  if(productionUpdatedAt) productionUpdatedAt.textContent='--';

  try{
    await startCamera();
    await apiPost('/api/production/push/start',{
      dataset:currentDataset,
      model_name:modelName,
      interval_ms:productionDetectIntervalMs,
      camera_id:1
    });

    if(productionStatus) productionStatus.textContent='连续检测中';
    await refreshProductionPushStatus();

    if(productionTimer){clearInterval(productionTimer);}
    productionTimer=setInterval(refreshProductionPushStatus, productionPollIntervalMs);
  }catch(err){
    productionRunning=false;
    if(productionStatus) productionStatus.textContent='启动失败';
    if(productionResultSummary) productionResultSummary.textContent=err.message || '生产连续检测启动失败';
    updateCameraLifecycle();
  }
}

function stopProductionMonitor(){
  if(productionTimer){
    clearInterval(productionTimer);
    productionTimer=null;
  }

  const wasRunning=productionRunning;
  productionRunning=false;
  productionBusy=false;
  productionLastPredictions=[];

  apiPost('/api/production/push/stop').catch(()=>{});

  if(productionStatus && wasRunning) productionStatus.textContent='已停止';
  updateCameraLifecycle();
}

async function initApp(){
  try{await loadCollectorState();await loadModels();await loadValidationImages();await refreshVisionBoxEffectiveStatus();showToast('本地采集目录已就绪');}catch(err){showToast(err.message)}
  updateCameraLifecycle();
  startCppStatusPolling();
}
initApp();


// 设置弹窗前端交互：v2.1.1 保存/回显 + RTSP/上传配置
const settingsModal=document.getElementById('settingsModal');
const openSettingsModalBtn=document.getElementById('openSettingsModal');
const closeSettingsModalBtn=document.getElementById('closeSettingsModal');
const settingTabs=document.querySelectorAll('.settings-main-tab');
const settingPanels=document.querySelectorAll('.settings-panel');
const settingCameraType=document.getElementById('settingCameraType');
const cameraTypePanels={
  usb:document.getElementById('cameraTypeUsb'),
  rtsp:document.getElementById('cameraTypeRtsp'),
  industrial:document.getElementById('cameraTypeIndustrial'),
  mock:document.getElementById('cameraTypeMock')
};
const usbDeviceNodeSelect=document.getElementById('usbDeviceNodeSelect');
const refreshUsbDevicesBtn=document.getElementById('refreshUsbDevicesBtn');
const usbDeviceHint=document.getElementById('usbDeviceHint');
const cppSettingCameraType=document.getElementById('cppSettingCameraType');
const cppCameraTypePanels={
  rtsp:document.getElementById('cppCameraTypeRtsp'),
  usb:document.getElementById('cppCameraTypeUsb'),
};
const cppUsbDeviceNodeSelect=document.getElementById('cppUsbDeviceNodeSelect');
const refreshCppUsbDevicesBtn=document.getElementById('refreshCppUsbDevicesBtn');
const cppUsbDeviceHint=document.getElementById('cppUsbDeviceHint');
const cppSettingResolution=document.getElementById('cppSettingResolution');
const cppSettingsStatus=document.getElementById('cppSettingsStatus');
const loadCppSettingsBtn=document.getElementById('loadCppSettingsBtn');
const saveCppSettingsBtn=document.getElementById('saveCppSettingsBtn');
const applyCppSettingsBtn=document.getElementById('applyCppSettingsBtn');
const testCppSettingsPreviewBtn=document.getElementById('testCppSettingsPreviewBtn');
let usbDevicesLoaded=false;
let usbDevicesLoading=false;
let cppUsbDevicesLoaded=false;
let cppUsbDevicesLoading=false;
let runtimeSettingsCache=null;
let cppSettingsCache=null;

function openSettingsModal(){
  if(settingsModal) settingsModal.classList.add('active');
  loadRuntimeSettingsToForm();
  loadCppSettingsToForm().catch(()=>{});
  refreshTimeSyncStatus();
}
function closeSettingsModal(){
  if(settingsModal) settingsModal.classList.remove('active');
}
function switchSettingsTab(tabId){
  settingTabs.forEach(btn=>btn.classList.toggle('active',btn.dataset.settingsTab===tabId));
  settingPanels.forEach(panel=>panel.classList.toggle('active',panel.id===tabId));
  if(tabId==='cppSettingsPanel'){
    loadCppSettingsToForm().catch(()=>{});
    if((cppSettingCameraType && cppSettingCameraType.value)==='usb' && !cppUsbDevicesLoaded){
      refreshCppUsbCameraDevices().catch(()=>{});
    }
  }
}
function updateCameraTypePanel(){
  const type=(settingCameraType && settingCameraType.value) || 'rtsp';
  Object.entries(cameraTypePanels).forEach(([key,panel])=>{
    if(panel) panel.classList.toggle('active',key===type);
  });
  if(type==='usb' && !usbDevicesLoaded) refreshUsbCameraDevices().catch(()=>{});
}
function getByPath(obj,path){
  return path.split('.').reduce((cur,key)=>{
    if(cur===undefined || cur===null) return undefined;
    return cur[key];
  },obj);
}
function setByPath(obj,path,value){
  const parts=path.split('.');
  let cur=obj;
  for(let i=0;i<parts.length-1;i++){
    const key=parts[i];
    if(!cur[key] || typeof cur[key] !== 'object') cur[key]={};
    cur=cur[key];
  }
  cur[parts[parts.length-1]]=value;
}
function parseSettingValue(el){
  if(el.type==='checkbox') return !!el.checked;
  if(el.tagName==='SELECT'){
    if(el.value==='true') return true;
    if(el.value==='false') return false;
    return el.value;
  }
  if(el.type==='number'){
    if(el.value==='') return null;
    const n=Number(el.value);
    return Number.isNaN(n)?el.value:n;
  }
  return el.value;
}
function writeSettingValue(el,value){
  if(value===undefined || value===null) return;
  if(el.type==='checkbox'){
    el.checked=!!value;
    return;
  }
  if(el.tagName==='SELECT'){
    const text=String(value);
    if(text && !Array.from(el.options).some(opt=>opt.value===text)){
      el.appendChild(new Option(text,text));
    }
  }
  el.value=String(value);
}
function collectSettingsFromForm(){
  const settings=runtimeSettingsCache?JSON.parse(JSON.stringify(runtimeSettingsCache)):{version:'2.3.0'};
  document.querySelectorAll('[data-setting]').forEach(el=>{
    const path=el.dataset.setting;
    if(!path) return;
    setByPath(settings,path,parseSettingValue(el));
  });
  settings.version='2.3.0';
  return settings;
}
function clampNumber(value, fallback, minValue, maxValue){
  let n=Number(value);
  if(!Number.isFinite(n)) n=fallback;
  if(Number.isFinite(minValue)) n=Math.max(minValue,n);
  if(Number.isFinite(maxValue)) n=Math.min(maxValue,n);
  return n;
}
function applyRuntimeSettingsToClient(settings){
  const common=(settings && settings.algorithm && settings.algorithm.common) || {};
  realtimeIntervalMs=clampNumber(common.realtime_interval_ms,1000,100,60000);
  const fpsLimit=clampNumber(common.production_fps_limit,5,1,30);
  const intervalByFps=Math.round(1000 / Math.max(1,fpsLimit));
  productionDetectIntervalMs=clampNumber(intervalByFps,intervalByFps,100,60000);
  productionPollIntervalMs=Math.max(300, Math.min(5000, productionDetectIntervalMs));

  if(realtimeRunning && realtimeTimer){
    clearInterval(realtimeTimer);
    realtimeTimer=setInterval(realtimeClassifyOnce,realtimeIntervalMs);
  }
  if(productionRunning && productionTimer){
    clearInterval(productionTimer);
    productionTimer=setInterval(refreshProductionPushStatus,productionPollIntervalMs);
  }
}
async function applyAlgorithmSettingsAfterSave(){
  const data=await apiPost('/api/settings/algorithm/apply',{});
  return data;
}

function formatUsbDeviceOption(item){
  const value=item.preferred_path || item.stable_path || item.path || '';
  const path=item.path || value;
  const title=item.name || item.card || path;
  const parts=[];
  if(item.recommended) parts.push('推荐');
  if(item.orbbec) parts.push('Orbbec');
  if(item.readable) parts.push('可读');
  if(item.width && item.height) parts.push(`${item.width}x${item.height}`);
  if(item.formats && item.formats.length) parts.push(item.formats.slice(0,3).join('/'));
  const suffix=parts.length ? `（${parts.join('，')}）` : '';
  return {value, label:`${value}${value!==path ? ` → ${path}` : ''} ${title}${suffix}`};
}

async function refreshUsbCameraDevices(preferredValue){
  if(!usbDeviceNodeSelect || usbDevicesLoading) return;
  usbDevicesLoading=true;
  if(usbDeviceHint) usbDeviceHint.textContent='正在扫描 USB 摄像头节点...';
  try{
    const data=await apiGet('/api/settings/camera/devices');
    const devices=(data && data.items) || [];
    const current=preferredValue || usbDeviceNodeSelect.value || getByPath(runtimeSettingsCache || {}, 'camera.usb.device_node') || '';
    usbDeviceNodeSelect.innerHTML='';

    if(!devices.length){
      const fallback=current || '/dev/video0';
      usbDeviceNodeSelect.appendChild(new Option(`${fallback}（未扫描到设备，请手动确认）`, fallback));
      usbDeviceNodeSelect.value=fallback;
      if(usbDeviceHint) usbDeviceHint.textContent='未扫描到 /dev/video*，请确认相机连接和 v4l-utils。';
      return;
    }

    let recommendedValue='';
    devices.forEach(item=>{
      const opt=formatUsbDeviceOption(item);
      if(!opt.value) return;
      const option=new Option(opt.label,opt.value);
      option.dataset.path=item.path || '';
      option.dataset.recommended=item.recommended ? '1' : '0';
      usbDeviceNodeSelect.appendChild(option);
      if(item.recommended && !recommendedValue) recommendedValue=opt.value;
    });

    const values=Array.from(usbDeviceNodeSelect.options).map(opt=>opt.value);
    if(current && !values.includes(current)){
      usbDeviceNodeSelect.appendChild(new Option(`${current}（当前配置）`, current));
    }

    const shouldAutoPick=!current || current==='/dev/video0' || !values.includes(current);
    usbDeviceNodeSelect.value=(shouldAutoPick && recommendedValue) ? recommendedValue : (current || recommendedValue || usbDeviceNodeSelect.options[0].value);

    const recommended=devices.find(item=>item.recommended);
    if(usbDeviceHint){
      usbDeviceHint.textContent=recommended
        ? `已推荐：${recommended.preferred_path || recommended.stable_path || recommended.path}；如测试失败，可改选其他节点。`
        : '已列出所有 /dev/video* 节点，未能自动判断 RGB 节点，请逐个测试可读项。';
    }
    usbDevicesLoaded=true;
  }catch(err){
    if(usbDeviceHint) usbDeviceHint.textContent=`扫描失败：${err.message || err}`;
  }finally{
    usbDevicesLoading=false;
  }
}


function updateCppCameraTypePanel(){
  const type=(cppSettingCameraType && cppSettingCameraType.value) || 'rtsp';
  Object.entries(cppCameraTypePanels).forEach(([key,panel])=>{
    if(panel) panel.classList.toggle('active', key===type);
  });
  if(type==='usb' && !cppUsbDevicesLoaded){
    refreshCppUsbCameraDevices().catch(()=>{});
  }
}
function setCppSettingsStatus(message, isWarn=false){
  if(!cppSettingsStatus) return;
  cppSettingsStatus.textContent=message || '--';
  cppSettingsStatus.classList.toggle('warn', !!isWarn);
}
function normalizeCppSettingsPayload(raw){
  const source=raw && typeof raw==='object' ? raw : {};
  const out={...source};
  const type=String(out.camera_type || 'rtsp').toLowerCase();
  out.camera_type=(type==='usb') ? 'usb' : 'rtsp';
  out.camera_source=String(out.camera_source || (out.camera_type==='usb' ? '/dev/video7' : '')).trim();
  out.stream_backend=String(out.stream_backend || 'opencv').trim() || 'opencv';
  out.rtsp_transport=String(out.rtsp_transport || 'tcp').toLowerCase()==='udp' ? 'udp' : 'tcp';
  out.rtsp_timeout_ms=Number(out.rtsp_timeout_ms || 5000);
  out.camera_width=Number(out.camera_width || 0);
  out.camera_height=Number(out.camera_height || 0);
  out.camera_fps=Number(out.camera_fps || (out.camera_type==='usb' ? 10 : 0));
  out.camera_buffer_size=Number(out.camera_buffer_size || 1);
  let fourcc=String(out.camera_fourcc || '').toUpperCase();
  if(fourcc==='MJPEG') fourcc='MJPG';
  out.camera_fourcc=out.camera_type==='usb' ? (fourcc || 'YUYV') : '';
  out.stream_auto_start=String(out.stream_auto_start)==='true' || out.stream_auto_start===true;
  out.camera_read_fps=Number(out.camera_read_fps || 10);
  out.detect_fps=Number(out.detect_fps || 10);
  out.snapshot_fps=Number(out.snapshot_fps || 5);
  out.preprocess_backend=String(out.preprocess_backend || 'auto');
  out.rga_mode=String(out.rga_mode || 'resize_color');
  return out;
}
function setCppSelectOrAppend(select,value){
  if(!select) return;
  const text=String(value ?? '');
  if(text && !Array.from(select.options).some(opt=>opt.value===text)){
    select.appendChild(new Option(text,text));
  }
  select.value=text;
}
function fillCppSettingsForm(settings){
  const cfg=normalizeCppSettingsPayload(settings || {});
  cppSettingsCache=cfg;
  applyCppPreviewFpsFromSettings(cfg,true);
  document.querySelectorAll('[data-cpp-setting]').forEach(el=>{
    const key=el.dataset.cppSetting;
    if(!key) return;
    let value=cfg[key];
    if(value===undefined || value===null) return;
    if(el.type==='checkbox'){
      el.checked=!!value;
    }else if(el.tagName==='SELECT'){
      setCppSelectOrAppend(el,value);
    }else{
      el.value=String(value);
    }
  });
  if(document.getElementById('cppSettingRtspUrl') && cfg.camera_type==='rtsp'){
    document.getElementById('cppSettingRtspUrl').value=cfg.camera_source || '';
  }
  if(cppUsbDeviceNodeSelect && cfg.camera_type==='usb'){
    setCppSelectOrAppend(cppUsbDeviceNodeSelect,cfg.camera_source || '/dev/video7');
  }
  if(cppSettingResolution){
    const res=(cfg.camera_width && cfg.camera_height) ? `${cfg.camera_width}x${cfg.camera_height}` : '1280x800';
    setCppSelectOrAppend(cppSettingResolution,res);
  }
  updateCppCameraTypePanel();
  if(cfg.camera_type==='usb'){
    refreshCppUsbCameraDevices(cfg.camera_source).catch(()=>{});
  }
  const intervalMs=getCppPreviewIntervalMs();
  setCppSettingsStatus(`已读取 C++ 设置：${cfg.camera_type} · ${cfg.camera_source || '--'} · ${cfg.camera_width || 0}×${cfg.camera_height || 0} · ${cfg.camera_fourcc || '--'}；前端预览约 ${(1000/intervalMs).toFixed(1)}fps`);
}
function collectCppSettingsFromForm(){
  const base=cppSettingsCache?JSON.parse(JSON.stringify(cppSettingsCache)):{};
  document.querySelectorAll('[data-cpp-setting]').forEach(el=>{
    const key=el.dataset.cppSetting;
    if(!key) return;
    base[key]=parseSettingValue(el);
  });
  if(cppSettingResolution){
    const [w,h]=String(cppSettingResolution.value || '').split('x').map(x=>Number(x));
    if(Number.isFinite(w) && Number.isFinite(h)){
      base.camera_width=w;
      base.camera_height=h;
    }
  }
  base.camera_type=(cppSettingCameraType && cppSettingCameraType.value) || base.camera_type || 'rtsp';
  if(base.camera_type==='usb'){
    base.camera_source=(cppUsbDeviceNodeSelect && cppUsbDeviceNodeSelect.value) || base.camera_source || '/dev/video7';
    base.stream_backend=(document.getElementById('cppSettingUsbBackend') && document.getElementById('cppSettingUsbBackend').value) || 'opencv';
  }else{
    base.camera_source=(document.getElementById('cppSettingRtspUrl') && document.getElementById('cppSettingRtspUrl').value) || base.camera_source || '';
    base.stream_backend=(document.getElementById('cppSettingRtspBackend') && document.getElementById('cppSettingRtspBackend').value) || base.stream_backend || 'opencv';
  }
  const normalized=normalizeCppSettingsPayload(base);
  if(normalized.camera_type==='rtsp'){
    normalized.camera_width=0;
    normalized.camera_height=0;
    normalized.camera_fps=0;
    normalized.camera_fourcc='';
  }else{
    normalized.stream_backend='opencv';
  }
  return normalized;
}
async function loadCppSettingsToForm(){
  if(!cppSettingsStatus) return;
  setCppSettingsStatus('正在读取 C++ 设置...');
  try{
    const data=await apiGet('/api/cpp/settings');
    const settings=data.settings || data.effective_settings || data.saved_settings || {};
    fillCppSettingsForm(settings);
  }catch(err){
    setCppSettingsStatus(`读取 C++ 设置失败：${err.message || err}`, true);
  }
}
async function saveCppSettingsOnly(){
  try{
    const payload=collectCppSettingsFromForm();
    setCppSettingsStatus('正在保存 C++ 设置...');
    const data=await apiPost('/api/cpp/settings',payload);
    const settings=data.settings || payload;
    fillCppSettingsForm(settings);
    showToast(data.message || 'C++ 设置已保存');
  }catch(err){
    setCppSettingsStatus(`保存 C++ 设置失败：${err.message || err}`, true);
    showToast(err.message || '保存 C++ 设置失败');
  }
}
async function applyCppSettingsAndRestart(){
  try{
    const payload=collectCppSettingsFromForm();
    setCppSettingsStatus('正在写入 cpp.env 并重启 C++ 服务...');
    const data=await apiPost('/api/cpp/settings/apply',payload);
    const settings=data.settings || payload;
    fillCppSettingsForm(settings);
    await refreshCppInferenceStatus(true);
    showToast(data.message || 'C++ 设置已应用并重启服务');
  }catch(err){
    setCppSettingsStatus(`应用 C++ 设置失败：${err.message || err}`, true);
    showToast(err.message || '应用 C++ 设置失败');
  }
}
async function testCppSettingsPreview(){
  try{
    await applyCppSettingsAndRestart();
    await startCppPreviewStream();
    await openCppPreviewModal();
    setCppSettingsStatus('C++ 预览已启动，请在预览窗口确认画面。');
  }catch(err){
    setCppSettingsStatus(`测试 C++ 预览失败：${err.message || err}`, true);
    showToast(err.message || '测试 C++ 预览失败');
  }
}
async function refreshCppUsbCameraDevices(preferredValue){
  if(!cppUsbDeviceNodeSelect || cppUsbDevicesLoading) return;
  cppUsbDevicesLoading=true;
  if(cppUsbDeviceHint) cppUsbDeviceHint.textContent='正在扫描 USB 摄像头节点...';
  try{
    const data=await apiGet('/api/settings/camera/devices');
    const devices=(data && data.items) || [];
    const current=preferredValue || cppUsbDeviceNodeSelect.value || (cppSettingsCache && cppSettingsCache.camera_source) || '';
    cppUsbDeviceNodeSelect.innerHTML='';
    if(!devices.length){
      const fallback=current || '/dev/video7';
      cppUsbDeviceNodeSelect.appendChild(new Option(`${fallback}（未扫描到设备，请手动确认）`, fallback));
      cppUsbDeviceNodeSelect.value=fallback;
      if(cppUsbDeviceHint) cppUsbDeviceHint.textContent='未扫描到 /dev/video*，请确认相机连接和 v4l-utils。';
      return;
    }
    let recommendedValue='';
    devices.forEach(item=>{
      const opt=formatUsbDeviceOption(item);
      if(!opt.value) return;
      const option=new Option(opt.label,opt.value);
      option.dataset.path=item.path || '';
      option.dataset.recommended=item.recommended ? '1' : '0';
      cppUsbDeviceNodeSelect.appendChild(option);
      if(item.recommended && !recommendedValue) recommendedValue=opt.value;
    });
    const values=Array.from(cppUsbDeviceNodeSelect.options).map(opt=>opt.value);
    if(current && !values.includes(current)){
      cppUsbDeviceNodeSelect.appendChild(new Option(`${current}（当前配置）`, current));
    }
    const shouldAutoPick=!current || current==='/dev/video0' || !values.includes(current);
    cppUsbDeviceNodeSelect.value=(shouldAutoPick && recommendedValue) ? recommendedValue : (current || recommendedValue || cppUsbDeviceNodeSelect.options[0].value);
    const recommended=devices.find(item=>item.recommended);
    if(cppUsbDeviceHint){
      cppUsbDeviceHint.textContent=recommended
        ? `已推荐：${recommended.preferred_path || recommended.stable_path || recommended.path}；如测试失败，可改选其他节点。`
        : '已列出所有 /dev/video* 节点，未能自动判断 RGB 节点，请逐个测试可读项。';
    }
    cppUsbDevicesLoaded=true;
  }catch(err){
    if(cppUsbDeviceHint) cppUsbDeviceHint.textContent=`扫描失败：${err.message || err}`;
  }finally{
    cppUsbDevicesLoading=false;
  }
}

function fillSettingsForm(settings){
  if(!settings) return;
  document.querySelectorAll('[data-setting]').forEach(el=>{
    const value=getByPath(settings,el.dataset.setting);
    writeSettingValue(el,value);
  });
  updateCameraTypePanel();
  if(((settings.camera || {}).type || '') === 'usb'){
    refreshUsbCameraDevices(getByPath(settings,'camera.usb.device_node')).catch(()=>{});
  }
  applyRuntimeSettingsToClient(settings);
  refreshVisionBoxEffectiveStatus();
}
async function loadRuntimeSettingsToForm(){
  try{
    const data=await apiGet('/api/settings');
    runtimeSettingsCache=data.settings || {};
    fillSettingsForm(runtimeSettingsCache);
  }catch(err){
    showToast(err.message || '读取设置失败');
  }
}

function buildRtspUrlFromForm(){
  const ipEl=document.querySelector('[data-setting="camera.rtsp.ip"]');
  const portEl=document.querySelector('[data-setting="camera.rtsp.port"]');
  const channelEl=document.querySelector('[data-setting="camera.rtsp.channel"]');
  const userEl=document.querySelector('[data-setting="camera.rtsp.username"]');
  const passEl=document.querySelector('[data-setting="camera.rtsp.password"]');
  const ip=(ipEl && ipEl.value || '').trim();
  if(!ip) return '';
  const port=(portEl && portEl.value || '554').trim() || '554';
  const channel=(channelEl && channelEl.value || '102').trim() || '102';
  const user=(userEl && userEl.value || 'admin').trim() || 'admin';
  const pass=(passEl && passEl.value || '').trim();
  return `rtsp://${user}:${pass}@${ip}:${port}/Streaming/Channels/${channel}`;
}
function syncRtspUrlFromFields(){
  const urlEl=document.querySelector('[data-setting="camera.rtsp.url"]');
  const url=buildRtspUrlFromForm();
  if(urlEl && url) urlEl.value=url;
}
async function applyCameraSettingsAfterSave(){
  const data=await apiPost('/api/settings/camera/apply',{});
  return data;
}

function refreshBackendCameraPreview(){
  backendCameraAvailable=true;
  useBackendCamera=true;
  useRealCamera=false;
  if(isCapturePageVisible()){
    cameraActive=false;
    startCamera();
    return;
  }
  if(backendCamera){
    backendCamera.removeAttribute('src');
    backendCamera.src='/api/camera/stream?t=' + Date.now();
    backendCamera.style.display='';
    backendCamera.classList.add('active','backend-active');
  }
  if(cameraVideo) cameraVideo.classList.remove('active');
  if(simulatedCamera) simulatedCamera.classList.remove('active');
}

async function saveRuntimeSettingsFromForm(){
  try{
    if(settingCameraType && settingCameraType.value === 'rtsp') syncRtspUrlFromFields();
    const settings=collectSettingsFromForm();
    const data=await apiPost('/api/settings',settings);
    runtimeSettingsCache=data.settings || settings;
    fillSettingsForm(runtimeSettingsCache);
    let message=data.message || '设置已保存';
    try{
      const algoResult=await applyAlgorithmSettingsAfterSave();
      if(algoResult && algoResult.message) message=algoResult.message;
    }catch(algoErr){
      message=`设置已保存，但算法应用失败：${algoErr.message || algoErr}`;
    }
    try{
      const vbResult=await apiPost('/api/settings/vision-box/apply',{});
      updateVisionBoxEffectiveStatus(vbResult);
      refreshTimeSyncStatus();
      if(vbResult && vbResult.vision_box){
        deviceId=vbResult.vision_box.device_id || deviceId;
        const uploadDevice=document.getElementById('uploadDeviceId');
        const uploadCustomer=document.getElementById('uploadCustomerId');
        if(uploadDevice) uploadDevice.value=deviceId;
        if(uploadCustomer && vbResult.vision_box.customer_id) uploadCustomer.value=vbResult.vision_box.customer_id;
      }
    }catch(vbErr){
      message=`设置已保存，但视觉盒子参数应用失败：${vbErr.message || vbErr}`;
    }
    try{
      const applyResult=await applyCameraSettingsAfterSave();
      refreshBackendCameraPreview();
      if(applyResult && applyResult.message && !message.includes('失败')) message='设置已保存，相机、算法和视觉盒子基础参数已应用';
    }catch(applyErr){
      message=`设置已保存，但相机应用失败：${applyErr.message || applyErr}`;
    }
    showToast(message);
  }catch(err){
    showToast(err.message || '保存设置失败');
  }
}
async function resetRuntimeSettings(){
  try{
    const data=await apiPost('/api/settings/reset',{});
    runtimeSettingsCache=data.settings || {};
    fillSettingsForm(runtimeSettingsCache);
    refreshTimeSyncStatus();
    try{
      await applyAlgorithmSettingsAfterSave();
    }catch(_){/* 恢复默认后的算法应用失败不影响回显 */}
    try{
      await applyCameraSettingsAfterSave();
      refreshBackendCameraPreview();
    }catch(_){/* 恢复默认后的相机应用失败不影响回显 */}
    showToast(data.message || '已恢复默认设置');
  }catch(err){
    showToast(err.message || '恢复默认设置失败');
  }
}


async function testCameraConnection(){
  try{
    if(settingCameraType && settingCameraType.value === 'rtsp') syncRtspUrlFromFields();
    const settings=collectSettingsFromForm();
    const saved=await apiPost('/api/settings',settings);
    runtimeSettingsCache=saved.settings || settings;
    fillSettingsForm(runtimeSettingsCache);
    const data=await apiPost('/api/settings/camera/test',{});
    refreshBackendCameraPreview();
    showToast(data.message || '相机连接测试通过');
  }catch(err){
    showToast(err.message || '相机连接测试失败');
  }
}

if(openSettingsModalBtn) openSettingsModalBtn.addEventListener('click',openSettingsModal);
if(closeSettingsModalBtn) closeSettingsModalBtn.addEventListener('click',closeSettingsModal);
if(settingsModal){
  settingsModal.addEventListener('click',(e)=>{
    if(e.target===settingsModal) closeSettingsModal();
  });
}
settingTabs.forEach(btn=>btn.addEventListener('click',()=>switchSettingsTab(btn.dataset.settingsTab)));
if(settingCameraType) settingCameraType.addEventListener('change',updateCameraTypePanel);
const saveSettingsBtn=document.getElementById('saveSettingsMock');
const resetSettingsBtn=document.getElementById('resetSettingsMock');
const testRtspCameraBtn=document.getElementById('testRtspCameraBtn');
const refreshTimeSyncStatusBtn=document.getElementById('refreshTimeSyncStatusBtn');
const testTimeSyncBtn=document.getElementById('testTimeSyncBtn');
if(saveSettingsBtn) saveSettingsBtn.addEventListener('click',saveRuntimeSettingsFromForm);
if(resetSettingsBtn) resetSettingsBtn.addEventListener('click',resetRuntimeSettings);
if(testRtspCameraBtn) testRtspCameraBtn.addEventListener('click',testCameraConnection);
if(refreshUsbDevicesBtn) refreshUsbDevicesBtn.addEventListener('click',()=>refreshUsbCameraDevices().catch(err=>showToast(err.message || 'USB 设备扫描失败')));
if(cppSettingCameraType) cppSettingCameraType.addEventListener('change',updateCppCameraTypePanel);
if(refreshCppUsbDevicesBtn) refreshCppUsbDevicesBtn.addEventListener('click',()=>refreshCppUsbCameraDevices().catch(err=>showToast(err.message || 'C++ USB 设备扫描失败')));
if(loadCppSettingsBtn) loadCppSettingsBtn.addEventListener('click',()=>loadCppSettingsToForm());
if(saveCppSettingsBtn) saveCppSettingsBtn.addEventListener('click',saveCppSettingsOnly);
if(applyCppSettingsBtn) applyCppSettingsBtn.addEventListener('click',applyCppSettingsAndRestart);
if(testCppSettingsPreviewBtn) testCppSettingsPreviewBtn.addEventListener('click',testCppSettingsPreview);
document.querySelectorAll('[data-cpp-setting="snapshot_fps"]').forEach(el=>{
  el.addEventListener('change',()=>{
    const payload=collectCppSettingsFromForm();
    const intervalMs=applyCppPreviewFpsFromSettings(payload,true);
    setCppSettingsStatus(`Snapshot FPS 已更新为 ${payload.snapshot_fps}，前端预览间隔 ${intervalMs}ms，保存并应用后 C++ 服务端也会同步生效。`);
  });
});
if(refreshTimeSyncStatusBtn) refreshTimeSyncStatusBtn.addEventListener('click',refreshTimeSyncStatus);
if(testTimeSyncBtn) testTimeSyncBtn.addEventListener('click',testTimeSync);
document.querySelectorAll('[data-setting^="camera.rtsp."]').forEach(el=>{
  if(el.dataset.setting !== 'camera.rtsp.url') el.addEventListener('change',syncRtspUrlFromFields);
});
updateCppCameraTypePanel();
updateCameraTypePanel();