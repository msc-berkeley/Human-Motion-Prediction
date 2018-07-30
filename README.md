# Human-Motion-Prediction
ls.m is the original RLS PAA method ;
ls_x1b1.m is RLS PAA tested on prediction pattern : (t,t+1,t+2) -> (t+1,t+2,t+3);
identifier_based.m is the identifier-based algorithm tested on prediction pattern: (t,t+1,t+2) -> (t+3,t+4,t+5);
id_update.m is the version of identifier_based.m that trained cell by cell and with time stamp (this part has been commented);
id_update_x1b1.m is the version of id algorithm that trained cell by cell and with time stamp, note the prediction pattern is :(t,t+1,t+2) -> (t+1,t+2,t+3);

note: as for other auxiliary files, please refer to head comment. 

recommend:
recommended demonstration of identifier-based algorithm, please run id_update.m; recommended demonstration of RLS-PAA algorithm please run ls.m
