(window.webpackJsonp_N_E=window.webpackJsonp_N_E||[]).push([[27],{"43Fe":function(e,t,r){"use strict";r.d(t,"d",(function(){return P})),r.d(t,"a",(function(){return J})),r.d(t,"e",(function(){return N})),r.d(t,"h",(function(){return D})),r.d(t,"g",(function(){return E})),r.d(t,"i",(function(){return z})),r.d(t,"f",(function(){return G})),r.d(t,"b",(function(){return W})),r.d(t,"c",(function(){return A}));var n=r("rg98"),c=r("vJKn"),a=r.n(c),i=r("z7pX"),o=r("cpVT"),u=r("5rFJ"),s=r("slJV"),p=r("+9zt"),l=r("Jzha");function d(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:3,t="/web-client-api/getCollections?count=".concat(e);return fetch(t).then((function(e){return e.json()})).catch((function(e){return e}))}var v=r("34em"),b=r("Gq/i"),f=r("eW/n"),m=r("XJ5b"),O=r("WHGu"),y=r("mjZG"),j=r("NOLZ"),h=a.a.mark(V),x=a.a.mark(X),w=a.a.mark(F),g=a.a.mark(H),_=a.a.mark(C),k=a.a.mark(K),S=a.a.mark(R);function I(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function B(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?I(Object(r),!0).forEach((function(t){Object(o.a)(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):I(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}var P=function(){return{type:j.tb}},J=function(){return{type:j.jb}},N=function(e,t,r){return{type:j.qb,id:e,url:t,position:r}},D=function(e,t){return{type:j.Ab,id:e,topic:t}},E=function(e){return{type:j.wb,topic:e}},z=function(e){return{type:j.yb,topic:e}},G=function(e){return{type:j.sb,id:e}},T={itemsById:{},topicSections:[],collectionSet:[],recentSaves:[],impressions:{}},W=function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:T,t=arguments.length>1?arguments[1]:void 0;switch(t.type){case j.lb:var r=t.topicSections;return B(B({},e),{},{topicSections:r});case j.wb:var n=t.topic,c=new Set([].concat(Object(i.a)(e.topicSections),[n]));return B(B({},e),{},{topicSections:Array.from(c)});case j.yb:var a=t.topic,u=e.topicSections.filter((function(e){return e.id!==a.id}));return B(B({},e),{},{topicSections:u});case j.xb:var s=t.topic,p=t.itemsById,l=t.items,d=B(B({},e.itemsById),p);return B(B({},e),{},Object(o.a)({itemsById:d},"".concat(s,"Topic"),l));case j.qb:var v=t.id,b=Z(e,v,"saving");return B(B({},e),{},{itemsById:b});case j.rb:var f=t.id,m=Z(e,f,"saved"),O=new Set([f].concat(Object(i.a)(e.recentSaves)));return B(B({},e),{},{itemsById:m,recentSaves:Array.from(O)});case j.pb:var y=t.id,h=Z(e,y,"unsaved");return B(B({},e),{},{itemsById:h});case j.Bb:var x=t.id,w=Z(e,x,"unsaved");return B(B({},e),{},{itemsById:w});case j.zb:var g=t.id,_=Z(e,g,"saved");return B(B({},e),{},{itemsById:_});case j.kb:var k=t.data;return B(B({},e),{},{collectionSet:k});case j.ob:var S=t.items,I=new Set([].concat(Object(i.a)(S),Object(i.a)(e.recentSaves)));return B(B({},e),{},{recentSaves:Array.from(I)});case j.sb:var P=t.id,J=B(B({},e.impressions),{},Object(o.a)({},P,!0));return B(B({},e),{},{impressions:J});case j.Sd:return B(B({},e),{},{impressions:{}});default:return e}};function Z(e,t,r){var n=e.itemsById,c=n[t];return B(B({},n),{},Object(o.a)({},t,B(B({},c),{},{save_status:r})))}var A=[Object(u.g)(j.tb,V),Object(u.g)(j.nb,X),Object(u.g)(j.jb,H),Object(u.g)(j.vb,F),Object(u.g)(j.wb,F),Object(u.g)(j.qb,C),Object(u.g)(j.Ab,K),Object(u.g)(j.wb,R),Object(u.g)(j.yb,R)],L=function(e,t){return e.home["".concat(t,"Topic")]},q=function(e){return e.home.topicSections};function V(){var e,t,r;return a.a.wrap((function(n){for(;;)switch(n.prev=n.next){case 0:e=s.a.getItem(y.y),t=e?JSON.parse(e):[],r=0;case 3:if(!(r<t.length)){n.next=9;break}return n.next=6,Object(u.c)({type:j.vb,topic:t[r]});case 6:r++,n.next=3;break;case 9:return n.next=11,Object(u.c)({type:j.lb,topicSections:t});case 11:case"end":return n.stop()}}),h)}function X(){var e,t,r,n;return a.a.wrap((function(c){for(;;)switch(c.prev=c.next){case 0:return c.prev=0,c.next=3,U({count:5,offset:0,state:"unread",sort:"newest"});case 3:if(e=c.sent,t=e.items,r=e.itemsById,!(n=e.error)){c.next=10;break}return c.next=10,Object(u.c)({type:j.mb,error:n});case 10:return c.next=12,Object(u.c)({type:j.ob,items:t,itemsById:r});case 12:c.next=19;break;case 14:return c.prev=14,c.t0=c.catch(0),console.log(c.t0),c.next=19,Object(u.c)({type:j.mb,error:c.t0});case 19:case"end":return c.stop()}}),x,null,[[0,14]])}function F(e){var t,r,n,c,i;return a.a.wrap((function(a){for(;;)switch(a.prev=a.next){case 0:return t=e.topic,a.prev=1,a.next=4,Object(u.e)(L,t.topic);case 4:if(!a.sent){a.next=7;break}return a.abrupt("return");case 7:return a.next=9,Y(t);case 9:if(r=a.sent,n=r.items,c=r.itemsById,!(i=r.error)){a.next=17;break}return a.next=16,Object(u.c)({type:j.ub,error:i});case 16:return a.abrupt("return",a.sent);case 17:return a.next=19,Object(u.c)({type:j.xb,topic:t.topic,items:n,itemsById:c});case 19:a.next=26;break;case 21:return a.prev=21,a.t0=a.catch(1),console.log("catch",a.t0),a.next=26,Object(u.c)({type:j.ub,error:a.t0});case 26:case"end":return a.stop()}}),w,null,[[1,21]])}function H(){var e,t,r;return a.a.wrap((function(n){for(;;)switch(n.prev=n.next){case 0:return n.prev=0,n.next=3,$({count:2});case 3:if(e=n.sent,t=e.data,!(r=e.error)){n.next=10;break}return n.next=9,Object(u.c)({type:j.ub,error:r});case 9:return n.abrupt("return",n.sent);case 10:return n.next=12,Object(u.c)({type:j.kb,data:t});case 12:n.next=19;break;case 14:return n.prev=14,n.t0=n.catch(0),console.log("catch",n.t0),n.next=19,Object(u.c)({type:j.ib,error:n.t0});case 19:case"end":return n.stop()}}),g,null,[[0,14]])}function C(e){var t,r,n,c,i,o,s,p;return a.a.wrap((function(a){for(;;)switch(a.prev=a.next){case 0:return t=e.url,r=e.id,n=e.position,a.prev=1,c={view:"web",section:"home",page:"/home/",cxt_item_position:n},a.next=5,Object(v.a)(t,c);case 5:if(1===(null===(i=a.sent)||void 0===i?void 0:i.status)){a.next=8;break}throw new Error("Unable to save");case 8:return a.next=10,Object(m.a)(Object.values(i.action_results));case 10:return o=a.sent,s=o.map((function(e){return e.resolved_id})),p=Object(O.a)(o,"resolved_id"),a.next=15,Object(u.c)({type:j.rb,id:r,items:s,itemsById:p});case 15:a.next=21;break;case 17:return a.prev=17,a.t0=a.catch(1),a.next=21,Object(u.c)({type:j.pb,error:a.t0});case 21:case"end":return a.stop()}}),_,null,[[1,17]])}function K(e){var t,r,n;return a.a.wrap((function(c){for(;;)switch(c.prev=c.next){case 0:return t=e.id,r=e.topic,c.prev=1,c.next=4,Object(b.a)(t);case 4:if(1===(null===(n=c.sent)||void 0===n?void 0:n.status)){c.next=7;break}throw new Error("Unable to remove item");case 7:return c.next=9,Object(u.c)({type:j.Bb,id:t,topic:r});case 9:c.next=15;break;case 11:return c.prev=11,c.t0=c.catch(1),c.next=15,Object(u.c)({type:j.zb,error:c.t0});case 15:case"end":return c.stop()}}),k,null,[[1,11]])}function R(e){var t;return a.a.wrap((function(r){for(;;)switch(r.prev=r.next){case 0:return e.topic,r.next=3,Object(u.e)(q);case 3:t=r.sent,s.a.setItem(y.y,JSON.stringify(t));case 5:case"end":return r.stop()}}),S)}function U(e){return M.apply(this,arguments)}function M(){return(M=Object(n.a)(a.a.mark((function e(t){var r,n,c,i,o;return a.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.prev=0,e.next=3,Object(p.a)(t);case 3:if((r=e.sent).list){e.next=6;break}return e.abrupt("return",{error:"No Items Returned"});case 6:return n=r.total,e.next=9,Object(m.a)(Object.values(r.list));case 9:return c=e.sent,i=c.sort((function(e,t){return e.sort_id-t.sort_id})).map((function(e){return e.resolved_id})),o=Object(O.a)(c,"resolved_id"),e.abrupt("return",{items:i,itemsById:o,total:n});case 15:e.prev=15,e.t0=e.catch(0),console.log("discover.state",e.t0);case 18:case"end":return e.stop()}}),e,null,[[0,15]])})))).apply(this,arguments)}function Y(e){return Q.apply(this,arguments)}function Q(){return(Q=Object(n.a)(a.a.mark((function e(t){var r,n,c,o,u,s,p,d;return a.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return r=t.topic,e.prev=1,e.next=4,Object(l.a)(r,12,0,!1);case 4:if((n=e.sent).curated){e.next=7;break}return e.abrupt("return",{error:"No Items Returned"});case 7:return c=n.curated,o=void 0===c?[]:c,e.next=10,Object(f.a)(o);case 10:return u=e.sent,s=u.map((function(e){return e.resolved_id})),p=Object(i.a)(new Set(s)),d=Object(O.a)(u,"resolved_id"),e.abrupt("return",{items:p,itemsById:d});case 17:e.prev=17,e.t0=e.catch(1),console.log("home.state.topics",e.t0);case 20:case"end":return e.stop()}}),e,null,[[1,17]])})))).apply(this,arguments)}function $(e){return ee.apply(this,arguments)}function ee(){return(ee=Object(n.a)(a.a.mark((function e(t){var r,n;return a.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return r=t.count,e.prev=1,e.next=4,d(r);case 4:if(!(n=e.sent).length){e.next=7;break}return e.abrupt("return",{data:n});case 7:return e.abrupt("return",{error:"No data found"});case 10:e.prev=10,e.t0=e.catch(1),console.log("home.state.topics",e.t0);case 13:case"end":return e.stop()}}),e,null,[[1,10]])})))).apply(this,arguments)}},IxLO:function(e,t,r){"use strict";r.d(t,"a",(function(){return b})),r.d(t,"b",(function(){return m})),r.d(t,"c",(function(){return O}));var n=r("rg98"),c=r("vJKn"),a=r.n(c),i=r("cpVT"),o=r("WHGu"),u=r("Jzha"),s=r("NOLZ"),p=r("5rFJ"),l=a.a.mark(y);function d(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function v(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?d(Object(r),!0).forEach((function(t){Object(i.a)(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):d(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}var b=function(){return{type:s.Xd}},f={activeTopic:"",topicsByName:{}},m=function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:f,t=arguments.length>1?arguments[1]:void 0;switch(t.type){case s.Wd:var r=t.hydrated;return v(v({},e),r);case s.Zd:var n=t.topicsByName;return v(v({},e),{},{topicsByName:n});case s.Yd:var c=t.topic;return v(v({},e),{},{activeTopic:c});case s.Cb:var a=t.payload.topicList;return v(v({},e),a);default:return e}},O=[Object(p.h)(s.Xd,y)];function y(e){var t;return a.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.prev=0,e.next=3,j();case 3:return t=e.sent,e.next=6,Object(p.c)({type:s.Zd,topicsByName:t});case 6:e.next=13;break;case 8:return e.prev=8,e.t0=e.catch(0),console.log(e.t0),e.next=13,Object(p.c)({type:s.Vd,error:e.t0});case 13:case"end":return e.stop()}}),l,null,[[0,8]])}function j(e){return h.apply(this,arguments)}function h(){return(h=Object(n.a)(a.a.mark((function e(t){var r,n,c;return a.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.prev=0,e.next=3,Object(u.b)(t);case 3:if((r=e.sent)&&1===r.status){e.next=6;break}return e.abrupt("return",{});case 6:return n=r.topics,c=Object(o.a)(n,"topic_slug"),e.abrupt("return",c);case 11:e.prev=11,e.t0=e.catch(0),console.log("topic-pages.topic-list.state",e.t0);case 14:case"end":return e.stop()}}),e,null,[[0,11]])})))).apply(this,arguments)}},Jzha:function(e,t,r){"use strict";r.d(t,"a",(function(){return c})),r.d(t,"b",(function(){return a}));var n=r("tv2I");function c(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:5,r=arguments.length>2&&void 0!==arguments[2]?arguments[2]:20,c=!(arguments.length>3&&void 0!==arguments[3])||arguments[3];return Object(n.a)({path:"v3/discover/topics",ssr:c,params:{topics:e,curated_count:t,algorithmic_count:r}})}function a(e){return Object(n.a)({ssr:e,path:"v3/discover/topicList"})}},"eW/n":function(e,t,r){"use strict";r.d(t,"a",(function(){return a}));var n=r("mjZG"),c=r("WHGu");function a(e){return e.map((function(e){var t;return{resolved_id:null===(t=e.item)||void 0===t?void 0:t.resolved_id,title:i(e),thumbnail:o(e),publisher:u(e),excerpt:s(e),save_url:l(e),open_url:p(e),read_time:d(e),syndicated:v(e),save_status:"unsaved"}}))}function i(e){var t=e.item,r=e.curated_info;return(null===r||void 0===r?void 0:r.title)||(null===t||void 0===t?void 0:t.title)||(null===t||void 0===t?void 0:t.resolved_title)||(null===t||void 0===t?void 0:t.given_title)||(null===t||void 0===t?void 0:t.display_url)||null}function o(e){var t,r,n=e.item,c=e.curated_info,a=(null===c||void 0===c?void 0:c.image_src)||(null===n||void 0===n?void 0:n.top_image_url)||(null===n||void 0===n||null===(t=n.images)||void 0===t||null===(r=t[Object.keys(n.images)[0]])||void 0===r?void 0:r.src)||null;return a||null}function u(e){var t,r,n,a=e.item,i=p({item:a}),o=Object(c.d)(i);return(null===a||void 0===a||null===(t=a.syndicated_article)||void 0===t||null===(r=t.publisher)||void 0===r?void 0:r.name)||(null===a||void 0===a||null===(n=a.domain_metadata)||void 0===n?void 0:n.name)||(null===a||void 0===a?void 0:a.domain)||o||null}function s(e){var t=e.item,r=e.curated_info;return(null===r||void 0===r?void 0:r.excerpt)||(null===t||void 0===t?void 0:t.excerpt)||null}function p(e){var t=e.item,r=e.redirect_url;return b(t)||r||(null===t||void 0===t?void 0:t.given_url)||(null===t||void 0===t?void 0:t.resolved_url)||null}function l(e){var t=e.item;return(null===t||void 0===t?void 0:t.given_url)||(null===t||void 0===t?void 0:t.resolved_url)||null}function d(e){var t,r=e.item;return(null===r||void 0===r?void 0:r.time_to_read)||!!(t=null===r||void 0===r?void 0:r.word_count)&&Math.ceil(parseInt(t,10)/n.O)||null}var v=function(e){var t=e.item;return!!t&&"syndicated_article"in t},b=function(e){var t=v({item:e}),r=(null===e||void 0===e?void 0:e.resolved_url)||!1;return!(!t||!r)&&"discover/item/".concat(r.substring(r.lastIndexOf("/")+1))}}}]);
//# sourceMappingURL=33f004c3602ae34a470050250261819e34abf824.a32f7dab2fc8cbf20bb4.js.map