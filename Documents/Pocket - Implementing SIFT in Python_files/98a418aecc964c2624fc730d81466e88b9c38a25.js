(window.webpackJsonp_N_E=window.webpackJsonp_N_E||[]).push([[7],{"2Idn":function(e,t,n){"use strict";function c(e){return(c="function"===typeof Symbol&&"symbol"===typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"===typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e})(e)}Object.defineProperty(t,"__esModule",{value:!0}),t.default=function(e){if(!("string"===typeof e||e instanceof String)){var t=c(e);throw null===e?t="null":"object"===t&&(t=e.constructor.name),new TypeError("Expected a string but received a ".concat(t))}},e.exports=t.default,e.exports.default=t.default},"2NWD":function(e,t,n){},"5AlR":function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),t.default=function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{},t=arguments.length>1?arguments[1]:void 0;for(var n in t)"undefined"===typeof e[n]&&(e[n]=t[n]);return e},e.exports=t.default,e.exports.default=t.default},"AR0+":function(e,t,n){"use strict";n.d(t,"a",(function(){return a}));var c=n("q1tI"),a=c.useLayoutEffect},GQvz:function(e,t,n){},J2Ia:function(e,t,n){},J75B:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),t.default=function(e){return(0,a.default)(e),e.replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/'/g,"&#x27;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/\//g,"&#x2F;").replace(/\\/g,"&#x5C;").replace(/`/g,"&#96;")};var c,a=(c=n("2Idn"))&&c.__esModule?c:{default:c};e.exports=t.default,e.exports.default=t.default},M8EM:function(e,t,n){},Oqnp:function(e,t,n){"use strict";var c=n("nKUr"),a=(n("q1tI"),n("TSYQ")),l=n.n(a),i=(n("T3N7"),n("UrOR")),r=function(e){var t=e.id,n=e.src,a=e.altText,r=e.size,o=e.className,s={width:r,height:r};return Object(c.jsx)("span",{className:l()("ajoca6s",{"with-image":!!n,default:!n},o),style:s,children:n?Object(c.jsx)("img",{src:n,alt:a,className:"i6op735","data-cy":"avatar-image-".concat(t)}):Object(c.jsx)(i.db,{className:"dwtruy8","data-cy":"avatar-default-".concat(t)})})};r.defaultProps={id:"",src:null,altText:"Your avatar"},t.a=r,n("M8EM")},U3VM:function(e,t,n){},XXKM:function(e,t,n){},Xdly:function(e,t,n){"use strict";var c=n("z7pX"),a=n("nKUr"),l=n("q1tI"),i=n.n(l),r=n("20a2"),o=n("/MKj"),s=n("A8Od"),u=n("Yxog"),d=n("T3N7"),b=n("TSYQ"),j=n.n(b),f=n("AR0+"),v=n("UrOR"),h=n("cpVT"),m=n("dhJC");function O(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var c=Object.getOwnPropertySymbols(e);t&&(c=c.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,c)}return n}function p(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?O(Object(n),!0).forEach((function(t){Object(h.a)(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):O(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}var g=function(e){var t=e.links,n=e.selectedLink,c=e.onLinkClick,l=e.className,i=Object(m.a)(e,["links","selectedLink","onLinkClick","className"]);return Object(a.jsx)("ul",p(p({className:Object(d.a)("l1fcr43o",l)},i),{},{children:t.map((function(e){var t=e.name===n;return Object(a.jsx)("li",{children:Object(a.jsx)("a",{id:e.id,className:t?"selected":"",href:e.url,onClick:function(t){var n,a;n=e.name,a=e.url,c(n,a)},children:e.label})},"global-nav-link-".concat(e.name))}))}))};g.defaultProps={links:[{name:"discover",id:"discover",label:Object(a.jsx)(s.Trans,{i18nKey:"nav:discover",children:"Discover"}),url:"https://getpocket.com/explore?src=navbar"},{name:"my-list",id:"my-list",label:Object(a.jsx)(s.Trans,{i18nKey:"nav:my-list",children:"My List"}),url:"https://getpocket.com/my-list?src=navbar"}],selectedLink:null,onLinkClick:function(e,t,n){}};var x=g;n("U3VM");var k=n("YFqc"),y=n.n(k),C=[{name:"upgrade",id:"upgrade-to-premium",label:Object(a.jsx)(s.Trans,{i18nKey:"nav:upgrade",children:"Upgrade"}),url:"https://getpocket.com/premium?src=navbar",icon:Object(a.jsx)(v.cb,{})}],N="i1oaez37",w=function(e){var t=e.link,n=t.name,c=t.isDisabled,l=void 0!==c&&c,i=t.url,r=t.icon,o=t.label,s=t.id,u=e.isSelected,d=e.handleClick;return Object(a.jsx)("li",{children:Object(a.jsx)(y.a,{href:l?null:i,children:Object(a.jsxs)("a",{id:s,className:j()({selected:u,disabled:l}),onClick:function(e){d(e,n,i)},children:[r||null,o]})})})},M=function(e){var t=e.handleClose,n=Object(s.useTranslation)().t;return Object(a.jsx)("div",{className:"dokzgub",children:Object(a.jsx)(v.h,{onClick:t,variant:"inline",className:N,children:Object(a.jsx)(v.k,{id:"mobile-menu-chevron-icon",title:n("nav:close","Close"),description:n("nav:close-the-pocket-mobile-menu","Close the Pocket mobile menu")})})})},_=function(e){var t=e.links,n=e.subLinks,c=e.selectedLink,l=e.isUserLoggedIn,i=e.isUserPremium,r=e.handleClick;return Object(a.jsxs)("ul",{className:"lg0uqr9",children:[t.map((function(e){var t=e.name===c,n="global-nav-mobile-menu-".concat(null===e||void 0===e?void 0:e.name);return Object(a.jsx)(w,{link:e,isSelected:t,handleClick:function(t){r(t,e.name,e.url)}},n)})),n?n.map((function(e){var t="global-nav-mobile-menu-".concat(null===e||void 0===e?void 0:e.name);return e.url?Object(a.jsx)(w,{link:e,handleClick:function(t){r(t,e.name,e.url)}},t):null})):null,l&&!i?Object(a.jsxs)(a.Fragment,{children:[Object(a.jsx)("hr",{className:"nav-divider"}),Object(a.jsx)("span",{className:"subhead","data-cy":"premium-nudge-section",children:Object(a.jsx)(s.Trans,{i18nKey:"nav:try-premium",children:"Try Premium"})}),C.map((function(e){var t=e.name===c;return Object(a.jsx)(w,{link:e,isSelected:t,handleClick:r},"global-nav-mobile-menu-".concat(null===e||void 0===e?void 0:e.name))}))]}):null]})},S=function(e){var t=e.links,n=e.subLinks,c=(e.subset,e.tag,e.selectedLink),l=e.onLinkClick,i=e.isUserLoggedIn,r=e.isUserPremium,o=e.onOpen,u=e.onClosed,d=e.appRootSelector,b=e.isOpen,f=e.toggleMenuOpen,h=e.toggleClass,m=Object(s.useTranslation)().t;var O=function(){u(),f(!1)};return Object(a.jsxs)(a.Fragment,{children:[Object(a.jsx)(v.h,{onClick:function(){o(),f(!0)},variant:"inline",className:j()(N,h),children:Object(a.jsx)(v.O,{id:"mobile-menu-menu-icon",title:m("nav:open","Open"),description:m("nav:open-the-pocket-mobile-menu","Open the Pocket mobile menu")})}),Object(a.jsxs)(v.s,{appRootSelector:d,isOpen:b,handleClose:O,screenReaderLabel:m("nav:pocket-mobile-menu","Pocket Mobile Menu"),children:[Object(a.jsx)(M,{handleClose:O}),Object(a.jsx)(_,{subLinks:n,links:t,selectedLink:c,handleClick:function(e,t,n){l(t,n)},isUserPremium:r,isUserLoggedIn:i})]})]})};S.defaultProps={links:[{name:"discover",id:"global-nav-discover-link",label:Object(a.jsx)(s.Trans,{i18nKey:"nav:discover",children:"Discover"}),url:"https://getpocket.com/explore?src=navbar",icon:Object(a.jsx)(v.r,{})},{name:"my-list",id:"global-nav-my-list-link",label:Object(a.jsx)(s.Trans,{i18nKey:"nav:my-list",children:"My List"}),url:"https://getpocket.com/my-list?src=navbar",icon:Object(a.jsx)(v.J,{})}],selectedLink:null,onLinkClick:function(e,t,n){},onOpen:function(){},onClosed:function(){},isUserLoggedIn:!1,isUserPremium:!1,forceShow:!1};var L=S;n("vjMB");var P=function(e){var t=e.tools,n=e.onToolClick;return t.length?Object(a.jsx)("ul",{className:"l1udqb06",children:t.map((function(e){return Object(a.jsx)("li",{children:Object(a.jsx)("button",{type:"button",title:e.label,onClick:function(t){var c;c=e.name,n(c)},children:e.icon})},"global-nav-tool-".concat(e.name))}))}):null};P.defaultProps={tools:[],onToolClick:function(e){}};var T=P;n("XXKM");var D=n("Oqnp");function A(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var c=Object.getOwnPropertySymbols(e);t&&(c=c.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,c)}return n}function E(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?A(Object(n),!0).forEach((function(t){Object(h.a)(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):A(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}var I=i.a.forwardRef((function(e,t){var n=e.id,c=e.size,l=e.label,i=e.src,r=e.onClick,o=e.className,s=Object(m.a)(e,["id","size","label","src","onClick","className"]);return Object(a.jsx)("button",E(E({type:"button",className:j()("b502wo9",o),title:l,onClick:r},s),{},{ref:t,children:Object(a.jsx)(D.a,{size:c,src:i,altText:l,"data-cy":"avatar-button-avatar-".concat(n)})}))}));I.defaultProps={label:"",id:"",src:null,onClick:function(){},className:null},n("xuf6");var U=n("iMuj"),q=function(e){var t=e.listMode,n=void 0===t?"grid":t,c=e.sortOrder,l=void 0===c?"newest":c,i=e.toggleSortOrder,r=e.setListMode,o=e.setGridMode,u=e.setDetailMode,b=Object(s.useTranslation)().t,j=function(e){return e===n};return Object(a.jsx)(v.ab,{children:Object(a.jsxs)("div",{className:"ldvpr6d",children:[Object(a.jsx)("div",{onClick:i,children:"newest"===l?Object(a.jsx)(v.xb,{label:b("settings:sort-items-by-oldest-first","Sort items by oldest first"),children:Object(a.jsx)("span",{className:"backing",children:Object(a.jsx)(v.mb,{})})}):Object(a.jsx)(v.xb,{label:b("settings:sort-items-by-newest-first","Sort items by newest first"),children:Object(a.jsx)("span",{className:"backing",children:Object(a.jsx)(v.lb,{})})})}),Object(a.jsx)("div",{className:Object(d.a)(j("list")&&"active"),onClick:r,children:Object(a.jsx)(v.xb,{label:b("settings:display-items-as-a-list","Display items as a list"),children:Object(a.jsx)("span",{className:"backing",children:Object(a.jsx)(v.J,{})})})}),Object(a.jsx)("div",{className:Object(d.a)(j("detail")&&"active"),onClick:u,children:Object(a.jsx)(v.xb,{label:b("settings:display-items-in-detail","Display items in detail"),children:Object(a.jsx)("span",{className:"backing",children:Object(a.jsx)(v.p,{})})})}),Object(a.jsx)("div",{className:Object(d.a)(j("grid")&&"active"),onClick:o,children:Object(a.jsx)(v.xb,{label:b("settings:display-items-as-a-grid","Display items as a grid"),children:Object(a.jsx)("span",{className:"backing",children:Object(a.jsx)(v.C,{})})})})]})})};n("rOzD");var K=n("Nj5D"),R="b1bq2cra",F=function(){return Object(a.jsx)("span",{"aria-hidden":"true",className:R})},z=function(){return Object(a.jsx)("span",{"aria-hidden":"true",className:j()(R,"f1806ge7")})};n("GQvz");var B="a6gcv5v",J=function(e){var t=e.isLoggedIn,n=e.isPremium,c=e.avatarSrc,i=e.accountName,r=e.profileUrl,o=e.appRootSelector,u=e.onLinkClick,d=e.onLoginClick,b=e.onAccountClick,j=e.userStatus,f=e.listMode,h=e.sortOrder,m=e.toggleSortOrder,O=e.toggleListMode,p=e.colorMode,g=e.setColorMode,x=e.setListMode,k=e.setGridMode,y=e.setDetailMode,C=e.sendImpression,N=e.showNotification,w=Object(s.useTranslation)().t,M=Object(l.useRef)(null);function _(e,t){u(e,t)}return"pending"===j?Object(a.jsx)(a.Fragment,{}):t?Object(a.jsxs)("div",{children:[n?null:Object(a.jsx)(K.a,{onVisible:function(){C("global-nav.upgrade-link")},children:Object(a.jsxs)("a",{href:"https://getpocket.com/premium?src=navbar",id:"global-nav.upgrade-link",className:"".concat(B," ").concat("u12stye0"),onClick:function(e){_("premium",e)},"data-cy":"upgrade-link",children:[Object(a.jsx)(v.cb,{}),Object(a.jsx)("span",{className:"label",children:Object(a.jsx)(s.Trans,{i18nKey:"nav:upgrade",children:"Upgrade"})})]})}),Object(a.jsxs)(v.xb,{label:w("nav:account","Account"),children:[Object(a.jsx)(I,{"aria-label":w("nav:open-account-menu","Open Account Menu"),src:c,ref:M,size:"40px",label:null,className:"a170u95e","data-cy":"account-menu-avatar"}),N?Object(a.jsx)(z,{"data-cy":"notification-avatar"}):null]}),Object(a.jsxs)(v.Z,{trigger:M,title:w("nav:account","Account"),screenReaderLabel:w("nav:account-menu","Account Menu"),appRootSelector:o,onOpen:b,popperOptions:{placement:"bottom-end",modifiers:[{name:"offset",options:{offset:[0,4]}}]},"data-cy":"account-menu",children:[Object(a.jsx)(v.ab,{children:Object(a.jsx)(v.bb,{helperText:w("nav:view-profile","View Profile"),href:r,id:"account-menu-profile-link",onClick:function(e){_("view-profile",e)},"data-cy":"account-menu-profile-link",children:i})}),Object(a.jsxs)(v.ab,{children:[Object(a.jsx)(v.bb,{href:"https://getpocket.com/options?src=navbar",id:"account-menu-manage-account-link",onClick:function(e){_("manage-account",e)},children:Object(a.jsx)(s.Trans,{i18nKey:"nav:manage-account",children:"Manage account"})}),Object(a.jsx)(v.bb,{href:"https://help.getpocket.com/category/847-category?src=navbar",id:"account-menu-help-link",onClick:function(e){_("help",e)},children:Object(a.jsx)(s.Trans,{i18nKey:"nav:get-help",children:"Get help"})}),Object(a.jsx)(v.bb,{href:"/my-list/messages",id:"account-menu-messages-link",onClick:function(e){_("messages",e)},children:Object(a.jsx)(s.Trans,{i18nKey:"nav:messages",children:"Messages"})}),Object(a.jsxs)(v.bb,{href:"/my-list/whats-new",id:"account-menu-whats-new-link",onClick:function(e){_("whats-new",e)},children:[Object(a.jsx)(s.Trans,{i18nKey:"nav:whats-new",children:"What\u2019s New"})," ",N?Object(a.jsx)(F,{"data-cy":"notification-whatsnew"}):null]})]}),Object(a.jsx)(v.ab,{children:Object(a.jsx)(v.bb,{href:"https://getpocket.com/lo?src=navbar",id:"account-menu-logout-link",onClick:function(e){_("logout",e)},children:Object(a.jsx)(s.Trans,{i18nKey:"nav:log-out",children:"Log out"})})}),Object(a.jsx)(U.a,{setColorMode:g,colorMode:p}),Object(a.jsx)(q,{listMode:f,sortOrder:h,toggleSortOrder:m,toggleListMode:O,setListMode:x,setGridMode:k,setDetailMode:y})]})]}):Object(a.jsxs)("div",{children:[Object(a.jsx)("a",{href:"https://getpocket.com/login?src=navbar",id:"global-nav-login-link",className:"".concat(B," login-link"),onClick:function(e){d(e)},"data-cy":"login-link",children:Object(a.jsx)(s.Trans,{i18nKey:"nav:log-in",children:"Log in"})}),Object(a.jsxs)(v.h,{href:"https://getpocket.com/signup?src=navbar",id:"global-nav-signup-link",className:"seuayju",variant:"secondary",onClick:function(e){_("signup",e)},"data-cy":"signup-link",children:[Object(a.jsx)(v.db,{}),Object(a.jsx)("span",{className:"label",children:Object(a.jsx)(s.Trans,{i18nKey:"nav:sign-up",children:"Sign up"})})]})]})};J.defaultProps={isLoggedIn:!1,isPremium:!1,avatarSrc:null,accountName:"You",profileUrl:null,onLinkClick:function(e){},onAccountClick:function(){}};var G=J;n("lTIE");s.Trans,v.r,s.Trans,v.J;var W=function(e){var t=e.subLinks,n=e.subset,c=e.tag,i=e.appRootSelector,r=e.pocketLogoOutboundUrl,o=e.selectedLink,u=e.isLoggedIn,d=e.isPremium,b=e.avatarSrc,h=e.accountName,m=e.profileUrl,O=e.onLinkClick,p=e.onToolClick,g=e.onLoginClick,k=e.onAccountClick,y=e.listMode,C=e.sortOrder,N=e.colorMode,w=e.setColorMode,M=e.toggleSortOrder,_=e.toggleListMode,S=e.setListMode,P=e.setGridMode,D=e.setDetailMode,A=e.showNotification,E=e.links,I=e.tools,U=e.sendImpression,q=e.userStatus,K=e.children,R=Object(s.useTranslation)().t,F=Object(l.useState)(!1),z=F[0],B=F[1],J=Object(v.Eb)(),W=J?J.width:v.Bb+1,$=Object(l.useState)(W<=v.Bb),H=$[0],V=$[1];return Object(f.a)((function(){V(W<=v.Bb)}),[W]),Object(a.jsx)("header",{className:j()("h1bh2prp",{"logged-in":u}),children:Object(a.jsx)(v.T,{className:"global-nav-container",children:Object(a.jsxs)("nav",{className:"n27eiag",children:[Object(a.jsxs)("div",{className:"site-nav",children:[Object(a.jsx)(L,{appRootSelector:i,links:E,subLinks:t,subset:n,tag:c,onLinkClick:O,selectedLink:o,toggleClass:"hamburger-icon",isOpen:z,toggleMenuOpen:B}),Object(a.jsxs)("a",{id:"pocket-logo-nav",className:"pocket-logo",href:r,onClick:function(e){!function(e,t){var n;if(H)return t.preventDefault(),void B(!0);var c=null===t||void 0===t||null===(n=t.currentTarget)||void 0===n?void 0:n.href;O(e,c)}("pocket",e)},"data-cy":"logo-link",children:[Object(a.jsx)(v.K,{className:"logo"}),u?Object(a.jsx)(v.L,{className:"logo-mark"}):null]})]}),K||Object(a.jsxs)(a.Fragment,{children:[Object(a.jsx)("div",{className:"lvjqdba","aria-label":R("nav:page-navigation","Page navigation"),children:Object(a.jsx)(x,{selectedLink:o,className:"links",links:E,onLinkClick:O,"data-cy":"primary-links"})}),Object(a.jsx)("div",{className:j()("tva5lsb",{"is-premium":d}),children:Object(a.jsx)(T,{tools:I,onToolClick:p})}),Object(a.jsx)(G,{onLoginClick:g,appRootSelector:i,isPremium:d,isLoggedIn:u,avatarSrc:b,accountName:h,profileUrl:m,onLinkClick:O,onAccountClick:k,userStatus:q,listMode:y,sortOrder:C,colorMode:N,setColorMode:w,toggleSortOrder:M,toggleListMode:_,sendImpression:U,setListMode:S,setGridMode:P,setDetailMode:D,showNotification:A})]})]})})})};W.defaultProps={pocketLogoOutboundUrl:"/explore?src=navbar",selectedLink:"",isLoggedIn:!1,isPremium:!1,avatarSrc:null,accountName:void 0,profileUrl:null,onLinkClick:function(e,t){},onToolClick:function(e,t){},onAccountClick:function(e){},tools:[],children:null};var $=W;n("gDWs");var H=n("mjZG"),V=n("imBb"),Q=n.n(V);function X(e){var t=e.searchTerms;return t.length?Object(a.jsxs)("div",{className:"r1yd3t3b",children:[Object(a.jsx)("h4",{className:"title",children:"Recent Searches"}),t.map((function(e){return Object(a.jsx)("div",{children:Object(a.jsx)(y.a,{href:"/my-list/search?query=".concat(e),children:Object(a.jsx)("a",{tabIndex:0,children:e})})},e)}))]}):null}n("j7Ux");var Y=function(e){var t=e.children;return Object(a.jsx)("span",{className:"c1nzshxn",children:t})},Z=function(e){var t=e.onClick;return Object(a.jsxs)("button",{className:"c1n69ads",onClick:t,children:[Object(a.jsx)(v.n,{className:"c14udmp5"}),Object(a.jsx)(Y,{children:Object(a.jsx)(s.Trans,{i18nKey:"nav:cancel",children:"Cancel"})})]})},ee=function(e){var t=e.onSubmit,n=e.onClose,c=e.onFocus,i=e.onBlur,r=e.value,o=e.placeholder,u=e.recentSearches,d=e.mobilePlaceholder,b=Object(s.useTranslation)().t,f=Object(l.useRef)(null),h=Object(l.useRef)(null),m=Object(l.useState)(r),O=m[0],p=m[1],g=Object(l.useState)(!1),x=g[0],k=g[1],y=Object(l.useState)(!1),C=y[0],N=y[1];return Object(l.useEffect)((function(){return Q.a.bind("esc",n),function(){return Q.a.unbind("esc")}}),[n]),Object(l.useEffect)((function(){k(window.innerWidth<v.Cb)}),[window.innerWidth]),Object(l.useLayoutEffect)((function(){if(u.length){var e='a, button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',t=h.current.querySelectorAll(e)[0],n=h.current.querySelectorAll(e),c=n[n.length-1],a=n.length-1,l=Array.from(n);return Q()(h.current).bind("up",(function(e){e.preventDefault();var t=l.indexOf(document.activeElement);n[0===t?a:t-1].focus()})),Q()(h.current).bind("down",(function(e){e.preventDefault();var t=l.indexOf(document.activeElement);n[t===a?0:t+1].focus()})),Q()(h.current).bind(["shift+tab"],(function(e){document.activeElement===t&&(e.preventDefault(),c.focus())})),Q()(h.current).bind(["tab"],(function(e){document.activeElement===c&&(e.preventDefault(),t.focus())})),t.focus(),function(){return Q.a.unbind(["shift+tab","up","tab","down"])}}}),[u]),Object(a.jsxs)("form",{className:"sxpxhia",onSubmit:function(e){if(e.stopPropagation(),e.preventDefault(),!O)return N(b("nav:please-enter-a-search-term"));t(O)},autoComplete:"off",ref:h,children:[Object(a.jsxs)("div",{className:"s1pj0kbg",children:[Object(a.jsx)(v.kb,{className:"szie4no"}),Object(a.jsx)("input",{name:"search-input",ref:f,className:j()(["search-input",{"has-value":!!O}]),"aria-label":b("nav:search-your-collection","Search your collection"),value:O,onChange:function(e){var t;N(!1),p(null===e||void 0===e||null===(t=e.target)||void 0===t?void 0:t.value)},onFocus:c,onBlur:i,onKeyUp:function(e){e.keyCode===H.J.ESCAPE&&f.current.blur()},placeholder:b(x?d:o),"data-cy":"search-input"}),Object(a.jsx)(X,{searchTerms:u}),C?Object(a.jsx)("div",{className:"error-message",children:Object(a.jsxs)("div",{children:[Object(a.jsx)(v.v,{})," ",C]})}):null]}),Object(a.jsx)("button",{className:"search-button","data-cy":"search-button",children:Object(a.jsx)(s.Trans,{i18nKey:"nav:search",children:"Search"})}),n?Object(a.jsx)(Z,{onClick:n,"data-cy":"search-close"}):null]})};ee.defaultProps={onClose:!1,onFocus:function(){},onBlur:function(){},value:"",placeholder:"nav:search-for-topics-and-interests",mobilePlaceholder:"nav:search-for-topics"};var te=ee;n("2NWD");var ne=n("ZAbA"),ce=n("fr1g"),ae=n("79MH"),le=n("puwg"),ie=n("J75B"),re=n.n(ie);var oe=function(e){var t=e.onClose,n=Object(r.useRouter)(),c=Object(o.b)(),i=Object(o.c)((function(e){var t;return 1===parseInt(null===e||void 0===e||null===(t=e.user)||void 0===t?void 0:t.premium_status,10)||!1})),s=Object(o.c)((function(e){var t;return null===e||void 0===e||null===(t=e.userSearch)||void 0===t?void 0:t.recent}));return Object(l.useEffect)((function(){c(Object(ne.a)())}),[c]),Object(a.jsx)(te,{recentSearches:s,onClose:t,onSubmit:function(e){c(function(e){return Object(ce.e)(ae.b,le.a,0,"global-nav.search.submit",e)}(e)),c(Object(ne.b)(e)),n.push("/my-list/search/?query=".concat(re()(e)))},isPremium:i})},se=n("bM/q"),ue=n.n(se),de=function(e){var t=e.children;return Object(a.jsx)("span",{className:"c1xd9hu9",children:t})},be=function(e){var t=e.onClick;return Object(a.jsxs)("button",{className:"cj9zxq3",onClick:t,children:[Object(a.jsx)(v.n,{className:"c6uzcx"}),Object(a.jsx)(de,{children:Object(a.jsx)(s.Trans,{i18nKey:"nav:cancel",children:"Cancel"})})]})},je=function(e){var t=e.onSubmit,n=e.onClose,c=e.onFocus,i=e.onBlur,r=e.value,o=e.placeholder,u=e.mobilePlaceholder,d=Object(s.useTranslation)().t,b=Object(l.useRef)(null),f=Object(l.useState)(r),h=f[0],m=f[1],O=Object(l.useState)(!1),p=O[0],g=O[1],x=Object(l.useState)(!1),k=x[0],y=x[1],C=function(e){if(e.stopPropagation(),e.preventDefault(),!ue()(h,{protocols:["http","https"],allow_underscores:!0}))return y(d("nav:please-enter-a-valid-url"));var c=new RegExp("^https?://").test(h)?"":"https://";t("".concat(c).concat(h)),n()};return Object(l.useEffect)((function(){return Q.a.bind("esc",n),function(){return Q.a.unbind("esc")}}),[n]),Object(l.useEffect)((function(){g(window.innerWidth<v.Cb)}),[window.innerWidth]),Object(l.useEffect)((function(){b.current.focus()}),[]),Object(a.jsxs)("form",{className:"a14hwmit",onSubmit:C,autoComplete:"off",children:[Object(a.jsxs)("div",{className:"a2ytnol",children:[Object(a.jsx)(v.b,{className:"a7o7dd7"}),Object(a.jsx)("input",{type:"url",name:"add-input",ref:b,className:j()(["add-input",{"has-value":!!h}]),"aria-label":d("nav:add-item-to-pocket","Add Item to Pocket"),value:h,onChange:function(e){var t;y(!1),m(null===e||void 0===e||null===(t=e.target)||void 0===t?void 0:t.value)},onFocus:c,onBlur:i,onKeyUp:function(e){e.keyCode===H.J.ESCAPE&&b.current.blur()},placeholder:d(p?u:o),"data-cy":"add-input"}),k?Object(a.jsx)("div",{className:"error-message",children:Object(a.jsxs)("div",{children:[Object(a.jsx)(v.v,{})," ",k]})}):null]}),Object(a.jsx)("button",{className:"add-button",onClick:C,"data-cy":"add-button",children:Object(a.jsx)(s.Trans,{i18nKey:"nav:add",children:"Add"})}),n?Object(a.jsx)(be,{onClick:n,"data-cy":"add-close"}):null]})};je.defaultProps={onClose:!1,onFocus:function(){},onBlur:function(){},value:"",placeholder:"nav:save-a-url-https",mobilePlaceholder:"nav:save-a-url"};var fe=je;n("mJ0U");var ve=n("c6BD");var he=function(e){var t=e.onClose,n=Object(o.b)();return Object(a.jsx)(fe,{onClose:t,onSubmit:function(e){n(function(e){return Object(ce.b)(ae.d,le.a,0,{save_url:e},"global-nav.save")}(e)),n(Object(ve.a)(e))}})},me="b172fsd5",Oe=function(e){var t=e.children;return Object(a.jsx)("span",{className:"c1dmz8tx",children:t})},pe="b1vi9mhm",ge=function(e){var t=e.onClick;return Object(a.jsxs)("button",{className:"".concat(pe," cancel-button"),onClick:t,children:[Object(a.jsx)(v.n,{className:"nio91go"}),Object(a.jsx)(Oe,{children:Object(a.jsx)(s.Trans,{i18nKey:"nav:cancel",children:"Cancel"})})]})};var xe=function(e){var t=e.onClose,n=e.batchFavorite,c=e.batchStatus,i=e.tagAction,r=e.favoriteAction,o=e.archiveAction,u=e.deleteAction,d=e.clearBulkItems,b=e.bulkItemsCount,j=Object(s.useTranslation)().t,f="favorite"===n,h="archive"===c,m=b>=1?d:t,O=b>=1?j("nav:clear-copy","Clear"):j("nav:cancel-copy","Cancel");return Object(l.useEffect)((function(){return Q.a.bind("esc",m),function(){return Q.a.unbind("esc")}}),[m]),Object(a.jsx)("div",{className:"b1xsx9mu",children:Object(a.jsxs)("div",{className:"bn8myry",children:[Object(a.jsx)("div",{className:"bulk-container",children:Object(a.jsxs)("div",{className:"bulk-actions",children:[Object(a.jsx)(v.xb,{label:j("nav:tag","Tag"),children:Object(a.jsx)("button",{className:pe,onClick:i,children:Object(a.jsx)(v.nb,{className:me})})}),Object(a.jsx)(v.xb,{label:f?j("nav:favorite","Favorite"):j("nav:unfavorite","Unfavorite"),children:Object(a.jsx)("button",{className:pe,onClick:r,children:f?Object(a.jsx)(v.A,{className:me}):Object(a.jsx)(v.z,{className:me})})}),Object(a.jsx)(v.xb,{label:h?j("nav:archive-tooltip","Archive"):j("nav:add-tooltip","Add"),children:Object(a.jsx)("button",{className:pe,onClick:o,children:h?Object(a.jsx)(v.c,{className:me}):Object(a.jsx)(v.b,{className:me})})}),Object(a.jsx)(v.xb,{label:j("nav:delete","Delete"),children:Object(a.jsx)("button",{className:pe,onClick:u,children:Object(a.jsx)(v.o,{className:me})})}),Object(a.jsx)("div",{className:"labelText",children:b?"".concat(b," item").concat(b>1?"s":""):Object(a.jsx)(s.Trans,{i18nKey:"nav:select-items",children:"Select Items"})})]})}),Object(a.jsx)("button",{className:"bulk-button",onClick:m,"data-cy":"clear-button",children:O}),t?Object(a.jsx)(ge,{onClick:t,"data-cy":"add-close"}):null]})})};n("J2Ia");var ke=n("ascl"),ye=n("Eq5H"),Ce=n("x0FB"),Ne=n("5wie"),we=n("pwxj");var Me=function(e){var t=e.onClose,n=Object(o.b)(),c=Object(o.c)((function(e){var t;return null===e||void 0===e||null===(t=e.bulkEdit)||void 0===t?void 0:t.selected})),l=Object(o.c)((function(e){var t;return null===e||void 0===e||null===(t=e.bulkEdit)||void 0===t?void 0:t.batchFavorite})),i=Object(o.c)((function(e){var t;return null===e||void 0===e||null===(t=e.bulkEdit)||void 0===t?void 0:t.batchStatus})),r=c.length,s="archive"===i?ye.d:ye.h,u="favorite"===l?Ce.d:Ce.h;return Object(a.jsx)(xe,{onClose:t,batchFavorite:l,batchStatus:i,tagAction:function(){var e;r&&(n((e=c,Object(ce.b)(ae.b,le.a,0,e,"global-nav.bulk.tag"))),n(Object(we.c)(c)))},favoriteAction:function(){r&&(n(function(e,t){var n=t?"global-nav.bulk.favorite":"global-nav.bulk.un-favorite";return Object(ce.b)(ae.b,le.a,0,e,n)}(c,"favorite"===l)),n(u(c)))},archiveAction:function(){r&&(n(function(e,t){var n=t?"global-nav.bulk.archive":"global-nav.bulk.un-archive",c=t?ae.d:ae.b;return Object(ce.b)(c,le.a,0,e,n)}(c,"archive"===i)),n(s(c)))},deleteAction:function(){var e;r&&(n((e=c,Object(ce.b)(ae.b,le.a,0,e,"global-nav.bulk.delete"))),n(Object(Ne.c)(c)))},clearBulkItems:function(){return n(Object(ke.c)())},bulkItemsCount:r})},_e=n("WHGu");t.a=function(e){var t,n=e.selectedLink,l=e.subset,i=e.tag,d=Object(s.useTranslation)().t,b=Object(o.b)(),j=Object(r.useRouter)(),f=void 0!==n?n:Object(_e.j)(j.pathname),h=Object(o.c)((function(e){var t;return null===e||void 0===e||null===(t=e.app)||void 0===t?void 0:t.mode})),m=Object(o.c)((function(e){var t;return null===e||void 0===e||null===(t=e.user)||void 0===t?void 0:t.user_status})),O=Object(o.c)((function(e){var t;return 1===parseInt(null===e||void 0===e||null===(t=e.user)||void 0===t?void 0:t.premium_status,10)||!1})),p=Object(o.c)((function(e){return!!e.user.auth})),g=function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:"";return["profileBlue.png"].reduce((function(t,n){return!e||(!1===t?t:e.includes(n))}),!0)?"":e}(Object(o.c)((function(e){var t,n;return null===e||void 0===e||null===(t=e.user)||void 0===t||null===(n=t.profile)||void 0===n?void 0:n.avatar_url}))),x=Object(o.c)((function(e){var t;return null===e||void 0===e||null===(t=e.user)||void 0===t?void 0:t.first_name})),k=Object(o.c)((function(e){var t;return null===e||void 0===e||null===(t=e.user)||void 0===t?void 0:t.user_id})),y="".concat(H.v,"/@").concat(k,"?src=navbar"),C=Object(o.c)((function(e){var t;return null===e||void 0===e||null===(t=e.app)||void 0===t?void 0:t.listMode})),N=Object(o.c)((function(e){var t;return null===e||void 0===e||null===(t=e.app)||void 0===t?void 0:t.sortOrder})),w=Object(o.c)((function(e){var t;return null===e||void 0===e||null===(t=e.app)||void 0===t?void 0:t.colorMode})),M=Object(o.c)((function(e){var t;return(null===e||void 0===e||null===(t=e.app)||void 0===t?void 0:t.releaseVersion)!==H.P})),_=Object(o.c)((function(e){return e.userTags.pinnedTags})).map((function(e){return{label:e,name:e,url:"/my-list/tags/".concat(e)}})),S=null===(t=Object(o.c)((function(e){return e.features["temp.web.client.home.new_user"]})))||void 0===t?void 0:t.assigned,L=[{name:"home",id:"global-nav-home-link",label:d("nav:home","Home"),url:"https://getpocket.com/home",icon:Object(a.jsx)(v.E,{})},{name:"my-list",id:"global-nav-my-list-link",label:d("nav:my-list","My List"),url:"/my-list",icon:Object(a.jsx)(v.J,{})},{name:"discover",id:"global-nav-discover-link",label:d("nav:discover","Discover"),url:"/explore",icon:Object(a.jsx)(v.r,{})}],P=[{name:"discover",id:"global-nav-discover-link",label:d("nav:discover","Discover"),url:"/explore",icon:Object(a.jsx)(v.r,{})},{name:"my-list",id:"global-nav-my-list-link",label:d("nav:my-list","My List"),url:"/my-list",icon:Object(a.jsx)(v.J,{})}],T=[{name:"archive",icon:Object(a.jsx)(v.c,{}),label:d("nav:archive","Archive"),url:"/my-list/archive"},{name:"favorites",icon:Object(a.jsx)(v.A,{}),label:d("nav:favorites","Favorites"),url:"/my-list/favorites"},{name:"highlights",icon:Object(a.jsx)(v.D,{}),label:d("nav:highlights","Highlights"),url:"/my-list/highlights"},{name:"articles",icon:Object(a.jsx)(v.f,{}),label:d("nav:articles","Articles"),url:"/my-list/articles"},{name:"videos",icon:Object(a.jsx)(v.ub,{}),label:d("nav:videos","Videos"),url:"/my-list/videos"},{name:"tags",icon:Object(a.jsx)(v.nb,{}),label:d("nav:all-tags","All Tags"),url:"/my-list/tags"}].concat(Object(c.a)(_)),D="my-list"===f&&p?[{name:"search",label:d("nav:search","Search"),icon:Object(a.jsx)(v.kb,{})},{name:"add-item",label:d("nav:save","Save a URL"),icon:Object(a.jsx)(v.b,{})},{name:"bulk-edit",label:d("nav:bulk-edit","Bulk Edit"),icon:Object(a.jsx)(v.t,{})}]:[],A={search:oe,add:he,bulk:Me}[h],E=S?L:P;return Object(a.jsx)($,{pocketLogoOutboundUrl:"/",appRootSelector:"#__next",links:E,subLinks:T,subset:l,tag:i,selectedLink:f,isLoggedIn:p,isPremium:O,avatarSrc:g,accountName:x,profileUrl:y,userStatus:m,onToolClick:function(e){var t;b((t="global-nav.".concat(e),Object(ce.e)(ae.b,le.a,0,t))),"search"===e&&b(Object(u.c)("search")),"add-item"===e&&b(Object(u.c)("add")),"bulk-edit"===e&&b(Object(u.c)("bulk"))},onLoginClick:function(e){e.preventDefault(),e.stopPropagation(),window.location.assign("".concat(H.L,"?src=navbar"))},listMode:C,sortOrder:N,toggleSortOrder:function(){return b(Object(u.r)())},colorMode:w,setColorMode:function(e){return b(Object(u.h)(e))},setListMode:function(){return b(Object(u.n)())},setGridMode:function(){return b(Object(u.m)())},setDetailMode:function(){return b(Object(u.l)())},toggleListMode:function(){return b(Object(u.g)())},sendImpression:function(e){return b(function(e){return Object(ce.f)(ae.f,ae.g,le.a,0,e)}(e))},showNotification:M,tools:D,children:A?Object(a.jsx)(A,{onClose:function(){return b(Object(u.c)("default"))}}):null})}},"bM/q":function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),t.default=function(e,t){if((0,c.default)(e),!e||/[\s<>]/.test(e))return!1;if(0===e.indexOf("mailto:"))return!1;if((t=(0,i.default)(t,o)).validate_length&&e.length>=2083)return!1;var n,r,d,b,j,f,v,h;if(v=e.split("#"),e=v.shift(),v=e.split("?"),e=v.shift(),(v=e.split("://")).length>1){if(n=v.shift().toLowerCase(),t.require_valid_protocol&&-1===t.protocols.indexOf(n))return!1}else{if(t.require_protocol)return!1;if("//"===e.substr(0,2)){if(!t.allow_protocol_relative_urls)return!1;v[0]=e.substr(2)}}if(""===(e=v.join("://")))return!1;if(v=e.split("/"),""===(e=v.shift())&&!t.require_host)return!0;if((v=e.split("@")).length>1){if(t.disallow_auth)return!1;if(-1===(r=v.shift()).indexOf(":")||r.indexOf(":")>=0&&r.split(":").length>2)return!1}b=v.join("@"),f=null,h=null;var m=b.match(s);m?(d="",h=m[1],f=m[2]||null):(v=b.split(":"),d=v.shift(),v.length&&(f=v.join(":")));if(null!==f){if(j=parseInt(f,10),!/^[0-9]+$/.test(f)||j<=0||j>65535)return!1}else if(t.require_port)return!1;if(!(0,l.default)(d)&&!(0,a.default)(d,t)&&(!h||!(0,l.default)(h,6)))return!1;if(d=d||h,t.host_whitelist&&!u(d,t.host_whitelist))return!1;if(t.host_blacklist&&u(d,t.host_blacklist))return!1;return!0};var c=r(n("2Idn")),a=r(n("f2Qg")),l=r(n("hHZz")),i=r(n("5AlR"));function r(e){return e&&e.__esModule?e:{default:e}}var o={protocols:["http","https","ftp"],require_tld:!0,require_protocol:!1,require_host:!0,require_port:!1,require_valid_protocol:!0,allow_underscores:!1,allow_trailing_dot:!1,allow_protocol_relative_urls:!1,validate_length:!0},s=/^\[([^\]]+)\](?::([0-9]+))?$/;function u(e,t){for(var n=0;n<t.length;n++){var c=t[n];if(e===c||(a=c,"[object RegExp]"===Object.prototype.toString.call(a)&&c.test(e)))return!0}var a;return!1}e.exports=t.default,e.exports.default=t.default},f2Qg:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),t.default=function(e,t){(0,c.default)(e),(t=(0,a.default)(t,i)).allow_trailing_dot&&"."===e[e.length-1]&&(e=e.substring(0,e.length-1));var n=e.split("."),l=n[n.length-1];if(t.require_tld){if(n.length<2)return!1;if(!/^([a-z\u00a1-\uffff]{2,}|xn[a-z0-9-]{2,})$/i.test(l))return!1;if(/[\s\u2002-\u200B\u202F\u205F\u3000\uFEFF\uDB40\uDC20\u00A9\uFFFD]/.test(l))return!1}if(!t.allow_numeric_tld&&/^\d+$/.test(l))return!1;return n.every((function(e){return!(e.length>63)&&(!!/^[a-z_\u00a1-\uffff0-9-]+$/i.test(e)&&(!/[\uff01-\uff5e]/.test(e)&&(!/^-|-$/.test(e)&&!(!t.allow_underscores&&/_/.test(e)))))}))};var c=l(n("2Idn")),a=l(n("5AlR"));function l(e){return e&&e.__esModule?e:{default:e}}var i={require_tld:!0,allow_underscores:!1,allow_trailing_dot:!1,allow_numeric_tld:!1};e.exports=t.default,e.exports.default=t.default},gDWs:function(e,t,n){},hHZz:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),t.default=function e(t){var n=arguments.length>1&&void 0!==arguments[1]?arguments[1]:"";if((0,a.default)(t),!(n=String(n)))return e(t,4)||e(t,6);if("4"===n){if(!l.test(t))return!1;var c=t.split(".").sort((function(e,t){return e-t}));return c[3]<=255}if("6"===n){var r=[t];if(t.includes("%")){if(2!==(r=t.split("%")).length)return!1;if(!r[0].includes(":"))return!1;if(""===r[1])return!1}var o=r[0].split(":"),s=!1,u=e(o[o.length-1],4),d=u?7:8;if(o.length>d)return!1;if("::"===t)return!0;"::"===t.substr(0,2)?(o.shift(),o.shift(),s=!0):"::"===t.substr(t.length-2)&&(o.pop(),o.pop(),s=!0);for(var b=0;b<o.length;++b)if(""===o[b]&&b>0&&b<o.length-1){if(s)return!1;s=!0}else if(u&&b===o.length-1);else if(!i.test(o[b]))return!1;return s?o.length>=1:o.length===d}return!1};var c,a=(c=n("2Idn"))&&c.__esModule?c:{default:c};var l=/^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$/,i=/^[0-9A-F]{1,4}$/i;e.exports=t.default,e.exports.default=t.default},j7Ux:function(e,t,n){},lTIE:function(e,t,n){},mJ0U:function(e,t,n){},rOzD:function(e,t,n){},vjMB:function(e,t,n){},xuf6:function(e,t,n){}}]);
//# sourceMappingURL=98a418aecc964c2624fc730d81466e88b9c38a25.1a3abcefbcb164f469bb.js.map