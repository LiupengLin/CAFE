clear;
clc;

%% landsat-8 and sentinel-2 list setting
dir_path = '..\train\250_500\';

landsat_list = dir(fullfile(dir_path,'*L7*.tif'));
modis_list = dir(fullfile(dir_path,'*M_*.tif'));

for para1=1:size(modis_list,1)
    modis_list_time(para1) = str2num(modis_list(para1).name(strfind(modis_list(para1).name,'M_')+2:strfind(modis_list(para1).name,'.tif')));
end

for para2=1:size(landsat_list,1)
    landsat_list_time(para2) = str2num(landsat_list(para2).name(strfind(landsat_list(para2).name,'L7_')+3:strfind(landsat_list(para2).name,'_30m')-1));
end

list1_len = size(modis_list_time,2);
list2_len = size(landsat_list_time,2);

for para3 =1:list1_len
    for para4=1:list2_len
        if(modis_list_time(para3)>landsat_list_time(para4))
            match_list(para3*2-2,1) = modis_list_time(para3);
            match_list(para3*2-2,2) = landsat_list_time(para4);
        elseif (modis_list_time(para3)<landsat_list_time(para4))
            match_list(para3*2-1,1) = modis_list_time(para3);
            match_list(para3*2-1,2) = landsat_list_time(para4);
            break;
        end
    end
end

match_list_len = size(match_list,1);

%% Data Serial Number
save_dir = '..\h5';
serial = 'cafe_250_500_20221017';
savepath = fullfile(save_dir,strcat('train_',serial,'.h5'));

%% Size Setting for Train Data 
patch_size = 40;
stride = 20;

%% Parameter Initialization

modis_tar_data_patch = zeros(patch_size,patch_size,6,1);
modis_ref_data_patch = zeros(patch_size,patch_size,6,1);
landsat_ref_data_patch = zeros(patch_size,patch_size,6,1);
landsat_tar_label_patch = zeros(patch_size,patch_size,6,1);

count = 0;

for para5 =1:match_list_len
    modis_tar_data = imread(fullfile(dir_path,strcat('M_',num2str(match_list(para5,1)),'.tif')));
    modis_ref_data = imread(fullfile(dir_path,strcat('M_',num2str(match_list(para5,2)),'.tif')));
    landsat_ref_data = imread(fullfile(dir_path,strcat('L7_',num2str(match_list(para5,2)),'_30m.tif')));
    landsat_tar_label = imread(fullfile(dir_path,strcat('L7_',num2str(match_list(para5,1)),'_30m.tif')));

    % Dataset generate 
    [hei,wid,cha] = size(landsat_tar_label);
    aa = floor((hei-patch_size)/stride)+1;
    bb = floor((wid-patch_size)/stride)+1;
 
    [hei,wid,cha] = size(landsat_tar_label);
    for parax = 1:1:floor((hei-patch_size)/stride)+1
        for paray = 1:1:floor((wid-patch_size)/stride)+1
            subim_modis_tar_data = modis_tar_data(1+(parax-1)*stride:patch_size+(parax-1)*stride,1+(paray-1)*stride:patch_size+(paray-1)*stride,:);
            subim_modis_ref_data = modis_ref_data(1+(parax-1)*stride:patch_size+(parax-1)*stride,1+(paray-1)*stride:patch_size+(paray-1)*stride,:);
            subim_landsat_ref_data = landsat_ref_data(1+(parax-1)*stride:patch_size+(parax-1)*stride,1+(paray-1)*stride:patch_size+(paray-1)*stride,:);
            subim_landsat_tar_label = landsat_tar_label(1+(parax-1)*stride:patch_size+(parax-1)*stride,1+(paray-1)*stride:patch_size+(paray-1)*stride,:);
            count=count+1;
            
            modis_tar_data_patch(:, :, :, count) = subim_modis_tar_data;
            modis_ref_data_patch(:, :, :, count) = subim_modis_ref_data;            
            landsat_ref_data_patch(:, :, :, count) = subim_landsat_ref_data;
            landsat_tar_label_patch(:, :, :, count) = subim_landsat_tar_label;
        end
    end
end

order = randperm(count);

modis_tar_data_patch = modis_tar_data_patch(:, :, :, order);
modis_ref_data_patch = modis_ref_data_patch(:, :, :, order);
landsat_ref_data_patch = landsat_ref_data_patch(:, :, :, order);
landsat_tar_label_patch = landsat_tar_label_patch(:, :, :, order); 

%% writing to HDF5
chunksz = 128;
created_flag = false;
totalct = 0;
for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata1 = modis_tar_data_patch(:,:,:,last_read+1:last_read+chunksz);
    batchdata2 = modis_ref_data_patch(:,:,:,last_read+1:last_read+chunksz);
    batchdata3 = landsat_ref_data_patch(:,:,:,last_read+1:last_read+chunksz);
    batchlabs = landsat_tar_label_patch(:,:,:,last_read+1:last_read+chunksz);
    startloc = struct('dat1',[1,1,1,totalct+1], 'dat2',[1,1,1,totalct+1], 'dat3',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = cafe_store2hdf5(savepath, batchdata1, batchdata2, batchdata3, batchlabs, ~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);