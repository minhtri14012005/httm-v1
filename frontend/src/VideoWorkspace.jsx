import React, {useEffect, useRef, useState} from 'react';
import {Camera, ChevronRight, Download, Play, Square, Trash2, Upload, Video} from 'lucide-react';

const jobStates={running:'Đang phân tích',stopping:'Đang dừng',completed:'Hoàn thành',cancelled:'Đã dừng',failed:'Gặp lỗi',interrupted:'Bị gián đoạn'};
const stamp=value=>value==null?'—':`${Math.floor(value/60).toString().padStart(2,'0')}:${(value%60).toFixed(1).padStart(4,'0')}`;
const validVideo=file=>!!file && file.size>0 && (file.type.startsWith('video/')||/\.(mp4|m4v|mov|mkv|avi|webm|mpeg|mpg)$/i.test(file.name));

function useJobEvents(job) {
 const [events,setEvents]=useState([]),[error,setError]=useState(false);
 useEffect(()=>{
  if(!job?.id){setEvents([]);setError(false);return;}
  let cancelled=false;
  setEvents([]);
  async function refresh(){
   try{const res=await fetch(`/api/events?job_id=${encodeURIComponent(job.id)}`,{cache:'no-store'});if(!res.ok)throw Error();const data=await res.json();if(!cancelled){setEvents(data);setError(false);}}
   catch{if(!cancelled)setError(true);}
  }
  refresh();
  const timer=setInterval(refresh,2000);
  return()=>{cancelled=true;clearInterval(timer);};
 },[job?.id]);
 return {events,error};
}

function VideoJobView({job,previewUrl,previewFailed,onPreviewError,tracks,trackJob,people,connected,onStop,onResults,DetectionPanel}){
 const {events,error}=useJobEvents(job);
 const [streamFailed,setStreamFailed]=useState(false);
 useEffect(()=>setStreamFailed(false),[job?.id]);
 const running=job&&['running','stopping'].includes(job.status);
 return <div className="monitor-layout"><div className="monitor-main">
  <div className="monitor-video">
   {running&&!streamFailed?<img src={`/api/preview?session=${job.id}`} alt="Khung hình video đang phân tích" onError={()=>setStreamFailed(true)}/>
    :job?.output&&!running?<video key={job.id} controls src={`/api/jobs/${job.id}/video`}/>
    :previewUrl&&!previewFailed?<video key={previewUrl} controls src={previewUrl} onError={onPreviewError} aria-label="Xem trước video đã chọn"/>
    :<div className="monitor-placeholder"><Video size={40}/><strong>{running?'Đang chờ khung hình từ backend':previewFailed?'Không thể xem trước video':'Chưa chọn video'}</strong><span>{running?'Bảng sự kiện vẫn tiếp tục cập nhật.':previewFailed?'Trình duyệt có thể không hỗ trợ định dạng này; backend sẽ kiểm tra khi phân tích.':'Chọn một video mới để xem trước và bắt đầu phân tích.'}</span></div>}
  </div>
  <div className="session-strip"><span className={'dot '+(running?'green':'')}/><strong>{job?jobStates[job.status]||job.status:'Chưa có phiên phân tích'}</strong>{job&&<span className="session-metrics">{job.processed_frames} frame · {stamp(job.video_time)}{job.duration?` / ${stamp(job.duration)}`:''} · {job.runtime?.selected?.toUpperCase()||'—'}</span>}{running&&<button className="danger" onClick={onStop}><Square size={14}/>Dừng</button>}</div>
  {job?.progress!=null&&<div className="monitor-progress" role="progressbar" aria-label="Tiến độ phân tích" aria-valuenow={Math.round(job.progress)} aria-valuemin={0} aria-valuemax={100}><span style={{width:`${Math.max(0,Math.min(100,job.progress))}%`}}/></div>}
  {job?.error&&<div className="alert error" role="alert">{job.error}</div>}
  {job?.status==='completed'&&<div className="inline-hint">Phân tích hoàn tất. Video chú thích gồm các frame đã phân tích, không kèm âm thanh gốc.</div>}
  {error&&<div className="alert error" role="status">Không tải được sự kiện của phiên này. Đang thử lại.</div>}
 </div><DetectionPanel key={job?.id||'empty'} job={job} events={events} tracks={trackJob===job?.id?tracks:[]} people={people} connected={connected} onResults={()=>onResults(job?.id)}/></div>;
}

function UploadVideo({jobs,active,busy,ready,watched,uploadProgress,onUpload,onStop,onResults,openSaved,...viewProps}){
 const input=useRef(null),generation=useRef(0);
 const [file,setFile]=useState(null),[preview,setPreview]=useState(null),[fileError,setFileError]=useState(''),[previewFailed,setPreviewFailed]=useState(false),[jobId,setJobId]=useState('');
 useEffect(()=>()=>{generation.current+=1;},[]);
 useEffect(()=>{
  if(!file){setPreview(null);return;}
  const objectUrl=URL.createObjectURL(file);setPreview({file,url:objectUrl});
  return()=>URL.revokeObjectURL(objectUrl);
 },[file]);
 const job=jobs.find(j=>j.id===jobId)||(active?.id===jobId?active:null);
 const acceptFile=next=>{
  if(!next||busy)return;
  generation.current+=1;setJobId('');setPreviewFailed(false);
  if(!validVideo(next)){setFile(null);setFileError('Chọn tệp video có định dạng được hỗ trợ và dung lượng lớn hơn 0.');return;}
  setFileError('');setFile(next);
 };
 const start=async()=>{
  if(!validVideo(file))return;
  const request=generation.current;
  const result=await onUpload(file);
  if(result?.id&&request===generation.current)setJobId(result.id);
 };
 return <>
  {active?.kind==='video'&&active.id!==jobId&&<div className="video-active-notice" role="status"><span>Có tác vụ video đang {active.status==='stopping'?'dừng':'phân tích'}: <strong>{active.name}</strong>. Form vẫn sẵn sàng cho lần chọn tiếp theo.</span><button className="text-button" onClick={()=>openSaved(active.id)}>Xem tác vụ <ChevronRight size={15}/></button></div>}
  <div className="monitor-controls upload-controls video-upload-controls"><div className="file-picker" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();acceptFile(e.dataTransfer.files?.[0]);}}><Upload size={21}/><div><strong>{file?.name||'Chọn hoặc kéo thả video mới'}</strong><small>{file?`${(file.size/1048576).toFixed(1)} MB`:'Phân tích offline trên máy chạy backend'}</small></div><button className="secondary" onClick={()=>input.current?.click()} disabled={busy}>Chọn tệp</button><input ref={input} className="visually-hidden" type="file" accept="video/*,.mkv,.avi,.mov" aria-label="Chọn video" disabled={busy} onChange={e=>{acceptFile(e.target.files?.[0]);e.target.value='';}}/></div><button className="primary" disabled={busy||!!active||!ready||!validVideo(file)||!watched.length} onClick={start}><Play size={16}/>{uploadProgress!=null?`Đang tải ${uploadProgress}%`:'Bắt đầu phân tích'}</button></div>
  {fileError&&<div className="alert error" role="alert">{fileError}</div>}
  <VideoJobView job={job} previewUrl={preview?.file===file?preview.url:''} previewFailed={previewFailed} onPreviewError={()=>setPreviewFailed(true)} onStop={onStop} onResults={onResults} {...viewProps}/>
 </>;
}

export function SavedVideos({jobs,initialId,onResults,onStop,onDeleteVideo,...viewProps}){
 const [selected,setSelected]=useState(initialId||''),[confirm,setConfirm]=useState(null),[deleting,setDeleting]=useState(false),[deleteError,setDeleteError]=useState(''),[notice,setNotice]=useState('');
 const videos=jobs.filter(j=>j.kind==='video').sort((a,b)=>(b.created||0)-(a.created||0));
 const job=videos.find(j=>j.id===selected);
 const isRunning=item=>['running','stopping'].includes(item.status);
 const deleteConfirmed=async()=>{
  setDeleting(true);setDeleteError('');
  try{
   const result=await onDeleteVideo(confirm.all?null:confirm.id);
   if(confirm.all||selected===confirm.id)setSelected('');
   setConfirm(null);
   setNotice(`Đã xóa ${result.deleted} tác vụ.${result.untracked_inputs?' Một số video cũ chưa ghi đường dẫn bản sao tải lên nên bản sao đó vẫn còn trong dữ liệu hệ thống.':''}`);
  }catch(error){setDeleteError(error.message||'Không thể xóa video.');}
  finally{setDeleting(false);}
 };
 return <div className="saved-videos">
  {notice&&<div className="alert" role="status">{notice}</div>}
  {videos.some(isRunning)&&<div className="inline-hint">Video đang xử lý không thể xóa. Dừng hoặc đợi tác vụ hoàn tất trước khi xóa tất cả.</div>}
  <div className="saved-list"><div className="saved-list-heading"><div><strong>Video đã lưu</strong> <span>{videos.length} tác vụ</span></div><button className="danger" disabled={!videos.length||videos.some(isRunning)} title={videos.some(isRunning)?'Dừng tác vụ video trước khi xóa tất cả':''} onClick={()=>{setNotice('');setDeleteError('');setConfirm({all:true});}}><Trash2 size={15}/>Xóa tất cả</button></div>
   {!videos.length?<p>Chưa có video đã phân tích.</p>:videos.map(item=><div key={item.id} className={'saved-job '+(selected===item.id?'chosen':'')}><button className="saved-job-select" onClick={()=>setSelected(item.id)} aria-pressed={selected===item.id}><Video size={20}/><span><strong>{item.name}</strong><small>{item.created?new Date(item.created*1000).toLocaleString('vi-VN'):'Không rõ thời gian'} · {jobStates[item.status]||item.status}</small></span><ChevronRight size={16}/></button>{item.status==='completed'&&item.output?<a className="saved-job-download" href={`/api/jobs/${encodeURIComponent(item.id)}/video?download=1`} download aria-label={'Tải video đã phân tích '+item.name} title="Tải video đã phân tích"><Download size={17}/></a>:<button className="saved-job-download" disabled aria-label={'Chưa thể tải video '+item.name} title="Chỉ tải được video đã phân tích hoàn tất"><Download size={17}/></button>}<button className="saved-job-delete" aria-label={'Xóa video '+item.name} title={isRunning(item)?'Không thể xóa tác vụ đang xử lý':'Xóa video'} disabled={isRunning(item)} onClick={()=>{setNotice('');setDeleteError('');setConfirm({id:item.id,name:item.name});}}><Trash2 size={17}/></button></div>)}
  </div>
  {job&&!confirm?<VideoJobView job={job} onStop={onStop} onResults={onResults} {...viewProps}/>:!confirm&&<div className="saved-empty">Chọn một tác vụ để xem trạng thái, video chú thích và sự kiện đã lưu.</div>}
  {confirm&&<div className="modal-backdrop"><section className="modal" role="dialog" aria-modal="true" aria-labelledby="delete-video-title"><h2 id="delete-video-title">{confirm.all?'Xóa tất cả video đã lưu?':`Xóa ${confirm.name}?`}</h2><p>Thao tác này xóa tác vụ, kết quả, ảnh minh chứng và các tệp video mà hệ thống xác định được. Với video cũ, bản sao tải lên có thể còn trong dữ liệu hệ thống nếu đường dẫn chưa được ghi lại. File bạn chọn ban đầu trên máy không bị xóa.</p>{deleteError&&<div className="alert error" role="alert">{deleteError}</div>}<div className="modal-actions"><button className="secondary" autoFocus disabled={deleting} onClick={()=>setConfirm(null)}>Hủy</button><button className="danger" disabled={deleting} onClick={deleteConfirmed}><Trash2 size={15}/>{deleting?'Đang xóa…':'Xác nhận xóa'}</button></div></section></div>}
 </div>;
}

export default function VideoWorkspace({setMode,jobs,active,busy,ready,people,uploadProgress,onUpload,onDeleteVideo,onStop,onResults,onIdentities,tracks,trackJob,connected,DetectionPanel}){
 const [section,setSection]=useState('upload'),[savedId,setSavedId]=useState('');
 const watched=people.filter(p=>p.watched);
 const openSaved=id=>{setSavedId(id||'');setSection('saved');};
 const viewProps={tracks,trackJob,people,connected,DetectionPanel};
 return <>
  <div className="mode-tabs" role="tablist" aria-label="Chế độ giám sát"><button role="tab" aria-selected="false" onClick={()=>setMode('live')}><Camera size={17}/> Live Camera</button><button role="tab" aria-selected="true" className="chosen"><Video size={17}/> Video Upload</button></div>
  <div className="watch-summary"><div><strong>Đang tìm {watched.length} người</strong><span>{watched.length?watched.map(p=>p.name).join(' · '):'Chưa chọn người cần tìm'}</span></div><button className="secondary" onClick={onIdentities}>Thay đổi watchlist <ChevronRight size={15}/></button></div>
  <div className="video-section-tabs" role="tablist" aria-label="Video"><button role="tab" aria-selected={section==='upload'} className={section==='upload'?'chosen':''} onClick={()=>setSection('upload')}>Upload Video</button><button role="tab" aria-selected={section==='saved'} className={section==='saved'?'chosen':''} onClick={()=>openSaved('')}>Video đã lưu</button></div>
  {section==='upload'?<UploadVideo jobs={jobs} active={active} busy={busy} ready={ready} watched={watched} uploadProgress={uploadProgress} onUpload={onUpload} onStop={onStop} onResults={onResults} openSaved={openSaved} {...viewProps}/>:<SavedVideos jobs={jobs} initialId={savedId} onResults={onResults} onStop={onStop} onDeleteVideo={onDeleteVideo} {...viewProps}/>}
 </>;
}
