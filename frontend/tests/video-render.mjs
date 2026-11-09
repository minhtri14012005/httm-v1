import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {createServer} from 'vite';

const server=await createServer({optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true},appType:'custom'});
try {
  const {DetectionPanel}=await server.ssrLoadModule('/src/Monitor.jsx');
  const {default:VideoWorkspace}=await server.ssrLoadModule('/src/VideoWorkspace.jsx');
  const job={id:'video-fixture',kind:'video',status:'completed'};
  const event={id:'stable-event',job_id:job.id,person_id:'anonymous',name:'Anonymous fixture',track_id:1,confirmed_at:.5};
  const render=props=>renderToStaticMarkup(React.createElement(DetectionPanel,{job,people:[],tracks:[],connected:true,onResults(){},...props}));
  for (const state of [{...event},{...event,verification_status:'unconfirmed'},
                       {...event,ended_at:8,end_reason:'lost'}]) {
    const html=render({events:[state,{...event,id:'pending',name:'Pending fixture',confirmed_at:null},
                                      {...event,id:'other',job_id:'other-job',name:'Other fixture'}]});
    assert.equal((html.match(/class="detection-item /g)||[]).length,1);
    assert(html.includes('Anonymous fixture'));
    assert(!html.includes('Pending fixture')&&!html.includes('Other fixture'));
    assert(html.includes('Đã phát hiện'));
    assert(!html.includes('Mất dấu')&&!html.includes('Hết hạn'));
  }
  const empty=render({events:[],tracks:[{track_id:1,candidate_name:'Candidate fixture'}]});
  assert(!empty.includes('Candidate fixture'));
  assert.equal((empty.match(/class="detection-item /g)||[]).length,0);
  const workspace=renderToStaticMarkup(React.createElement(VideoWorkspace,{
    jobs:[{...job,name:'Old saved video',output:'videos/old.mp4'}],people:[],tracks:[],
    connected:true,ready:true,busy:false,DetectionPanel,onUpload(){},setMode(){},onResults(){}
  }));
  assert(!workspace.includes('/api/jobs/video-fixture/video'));
  assert(!workspace.includes('Old saved video'));
  assert(workspace.includes('Chưa chọn video'));
  console.log('Frontend video render checks passed: confirmed-only events, stable event row, no lost/candidate notices, empty upload with saved history.');
} finally {await server.close();}
