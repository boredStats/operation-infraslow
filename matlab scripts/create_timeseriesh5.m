h5create('timeseries_MEG.h5','/100307/1',[360 length(timeseries1)],'Datatype','double', ...
         'ChunkSize',[360 1018],'Deflate',9)

h5create('timeseries_MEG.h5','/100307/2',[360 length(timeseries2)],'Datatype','double', ...
         'ChunkSize',[360 1018],'Deflate',9)
     
h5create('timeseries_MEG.h5','/100307/3',[360 length(timeseries3)],'Datatype','double', ...
         'ChunkSize',[360 1018],'Deflate',9)
     
h5write('timeseries_MEG.h5', '/100307/1', timeseries1)
h5write('timeseries_MEG.h5', '/100307/2', timeseries2)
h5write('timeseries_MEG.h5', '/100307/3', timeseries3)

h5disp('timeseries_MEG.h5')