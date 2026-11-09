import React, {useEffect, useState} from 'react';
import {AlertCircle, Camera, ChevronRight, Monitor as MonitorIcon, Play, RefreshCw, Smartphone, Square, Users, Video} from 'lucide-react';
import VideoWorkspace from './VideoWorkspace.jsx';

const jobStates={running:'Đang phân tích',stopping:'Đang dừng',completed:'Hoàn thành',cancelled:'Đã dừng',failed:'Gặp lỗi',interrupted:'Bị gián đoạn'};
const stamp=value=>value==null?'—':`${Math.floor(value/60).toString().padStart(2,'0')}:${(value%60).toFixed(1).padStart(4,'0')}`;

export function DetectionPanel({job,events,tracks,people,connected,onResults}){
 const [selected,setSelected]=useState('');
 const items=events.filter(e=>e.job_id===job?.id&&e.id&&e.person_id&&e.confirmed_at!=null)
  .map(e=>({key:e.id,event:e})).sort((a,b)=>b.event.confirmed_at-a.event.confirmed_at);
 const current=items.find(item=>item.key===selected);
 const picture=item=>{
  const id=item.event.person_id;
  const photo=people.find(p=>p.id===id)?.photos?.[0];
  return item.event.evidence?`/api/events/${item.event.id}/image`:photo?`/api/photos/${photo.id}`:null;
 };
 const eventTime=value=>job?.kind==='live'&&value!=null
  ?new Date(((job.started_wall??job.created)+value)*1000).toLocaleTimeString('vi-VN')
  :stamp(value);
 return <section className="detection-panel" aria-label="Sự kiện phát hiện">
  <div className="detection-head"><div><span className="eyebrow">THEO PHIÊN ĐANG XEM</span><h2>Sự kiện phát hiện</h2></div><span className="event-count">{items.length}</span></div>
  {!connected&&<div className="realtime-warning" role="status"><AlertCircle size={15}/> Kết nối realtime gián đoạn. Dữ liệu đã xác nhận vẫn được tải lại định kỳ.</div>}
  <div className="detection-scroll" aria-live="polite">
   {!job?<div className="detection-empty"><Users size={28}/><strong>Chưa có phiên phân tích</strong><span>Bắt đầu Live Camera hoặc tải video để theo dõi kết quả tại đây.</span></div>:!items.length?<div className="detection-empty"><Users size={28}/><strong>Chưa có danh tính được xác nhận</strong><span>Kết quả chỉ xuất hiện khi có bằng chứng. Không thấy kết quả không có nghĩa là người đó vắng mặt.</span></div>:items.map(item=>{
    const e=item.event,img=picture(item);
    return <button type="button" key={item.key} className={'detection-item '+(selected===item.key?'active':'')} onClick={()=>setSelected(selected===item.key?'':item.key)} aria-expanded={selected===item.key}>
     <span className="detection-avatar">{img?<img src={img} alt=""/>:<Users size={22}/>}</span><span className="detection-info"><strong>{e.name}</strong><small>Track #{e.track_id} · {job.kind==='live'?'Live Camera':'Video Upload'}</small><span className="detection-times">Phát hiện {eventTime(e.confirmed_at)}</span></span><span className="event-state confirmed">Đã phát hiện</span>
    </button>;
   })}
  </div>
  {current&&<div className="detection-detail"><strong>Chi tiết phát hiện</strong><div><span>Phát hiện</span><b>{eventTime(current.event.confirmed_at)}</b></div><div><span>Nguồn</span><b>{job.kind==='live'?'Live Camera':'Video Upload'}</b></div><small>{job.kind==='video'?'Thời điểm tính từ đầu video gốc.':'Giờ của phiên Live.'} Track ID chỉ là mã theo dõi.</small>{current.event.evidence&&<a href={`/api/events/${current.event.id}/image`} target="_blank" rel="noreferrer">Mở ảnh minh chứng <ChevronRight size={14}/></a>}</div>}
  {job&&<button className="panel-link" onClick={onResults}>Xem lịch sử và kết quả <ChevronRight size={16}/></button>}
 </section>;
}

export default function Monitor({mode,setMode,job,jobs,events,tracks,trackJob,people,connected,ready,busy,active,uploadProgress,onUpload,onDeleteVideo,onStop,cameras,source,setSource,onCameraRefresh,onLiveStart,onIdentities,onResults}){
 const [previewError,setPreviewError]=useState(false);
 useEffect(()=>setPreviewError(false),[job?.id]);
 const watched=people.filter(p=>p.watched);
 const camera=cameras.find(c=>c.key===source);
 const running=job&&['running','stopping'].includes(job.status);
 if(mode==='video')return <VideoWorkspace setMode={setMode} jobs={jobs} events={events} tracks={tracks} trackJob={trackJob} people={people} connected={connected} ready={ready} busy={busy} active={active} uploadProgress={uploadProgress} onUpload={onUpload} onDeleteVideo={onDeleteVideo} onStop={onStop} onIdentities={onIdentities} onResults={onResults} DetectionPanel={DetectionPanel}/>;
 return <>
  <div className="mode-tabs" role="tablist" aria-label="Chế độ giám sát"><button role="tab" aria-selected="true" className="chosen"><Camera size={17}/> Live Camera</button><button role="tab" aria-selected="false" onClick={()=>setMode('video')}><Video size={17}/> Video Upload</button></div>
  <div className="watch-summary"><div><strong>Đang tìm {watched.length} người</strong><span>{watched.length?watched.map(p=>p.name).join(' · '):'Chưa chọn người cần tìm'}</span></div><button className="secondary" onClick={onIdentities}>Thay đổi watchlist <ChevronRight size={15}/></button></div>
  <div className="monitor-layout"><div className="monitor-main">
   <div className="monitor-controls"><div className="camera-sources" role="radiogroup" aria-label="Nguồn camera">{[['laptop','Webcam Laptop',MonitorIcon],['camo','Camo Camera',Smartphone]].map(([key,name,Icon])=>{const c=cameras.find(x=>x.key===key);return <button type="button" role="radio" aria-checked={source===key} key={key} className={source===key?'chosen':''} onClick={()=>setSource(key)}><Icon size={18}/><span><strong>{name}</strong><small>{c?.available?c.name:c?.error||'Chưa dò thiết bị'}</small></span></button>;})}</div><div className="control-actions"><button className="secondary" disabled={busy} onClick={onCameraRefresh} aria-label="Dò lại camera"><RefreshCw size={17}/></button><button className="primary" disabled={busy||!ready||!watched.length||!camera?.available||active?.kind==='video'} onClick={onLiveStart}><Play size={16}/>{active?.kind==='live'?'Chuyển nguồn / mở lại':'Bắt đầu nhận diện'}</button>{active?.kind==='live'&&<button className="danger" disabled={busy} onClick={onStop}><Square size={15}/>Dừng</button>}</div></div>
   <div className="monitor-video">{running&&!previewError?<img src={`/api/preview?session=${job.id}`} alt="Hình camera đã phân tích" onError={()=>setPreviewError(true)}/>:<div className="monitor-placeholder"><Camera size={40}/><strong>{running?'Đang chờ khung hình từ backend':'Camera chưa hoạt động'}</strong><span>{running?'Bảng sự kiện vẫn tiếp tục cập nhật.':'Chọn camera rồi bắt đầu nhận diện.'}</span></div>}</div>
   <div className="session-strip"><span className={'dot '+(running?'green':'')}/><strong>{job?jobStates[job.status]||job.status:'Chưa có phiên'}</strong>{job&&<span className="session-metrics">{job.processed_frames} frame · {stamp(job.video_time)}{job.duration?` / ${stamp(job.duration)}`:''} · {job.runtime?.selected?.toUpperCase()||'—'}</span>}</div>
   {job?.error&&<div className="alert error" role="alert">{job.error}</div>}
   {!camera?.available&&<div className="inline-hint">{camera?.error||'Camera chưa khả dụng. Dò lại hoặc kiểm tra nguồn đã chọn.'}</div>}
  </div><DetectionPanel job={job} events={events} tracks={trackJob===job?.id?tracks:[]} people={people} connected={connected} onResults={()=>onResults(job?.id)}/></div>
 </>;
}
