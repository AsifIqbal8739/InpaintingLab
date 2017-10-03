%% To get patches from multiple images for dictionary learning
clc; clear ; close all;

%% Data Stuff
n = 8;  % Patch size
Num_P = 600;   % # of Patches from each Image
Images = {'barbara.png','boat.png','elaine.bmp','lena.png','Peppers512.png'};  
ImSize = 256;   % Image Size to use

%% Inpainting Stuff
Im_Select = 3;
Im = im2double(imread(strcat('.\Images\',Images{Im_Select})));
if size(Im,2) ~= ImSize
  Im = imresize(Im,[ImSize,ImSize]);
end
Im_D = 1024;    % For 256x256 images with Distinct patches 
Mperc = 0.5;    % Percentage of pixels to be ZERO within each patch
nM = round(n^2*Mperc);
Mask = ones(n^2,ImSize^2/n^2);

for i = 1:Im_D
    Mask(randperm(n^2,nM),i) = 0;
end

Mask_Full = col2im(Mask,[n,n],[ImSize,ImSize],'distinct');
Im_InP = Im.*Mask_Full;    % Pixels Removed from image

%% Loading the Dictionary
load('Dict_KSVD');      % First Atom is DC
K = size(Dict,2);   % Number of Atoms

%% Generating sliced patches image and its mask
[Im_InP_P,Im_InP_Loc] = pic2patches(Im_InP,n); % sliding patches, and patch center
[Mask_P,Mask_Loc] = pic2patches(Mask_Full,n);

% Learning Sparse Codes from Dictionary
SpCode = zeros(K,size(Im_InP_P,2));
Sparsity = 20;
E_T = 0.01;

for i = 1:size(Im_InP_P,2)
    DD = Dict;  DD(Mask_P(:,i)==0,:) = 0;   
    ssig = Im_InP_P(:,i);
    SpCode(:,i) = OMPerr(DD,ssig,E_T,Sparsity);
end

Im_InP_Rec = Dict * SpCode;
[Im_Rec,Seen] = patches2pic(Im_InP_Rec,Im_InP_Loc,n);

OutPSNR = psnr(Im_Rec,Im);
OutSSE = norm(Im_Rec(:)-Im(:));

% Display
figure;
subplot(1,3,1); imagesc(Im);colormap gray; axis image;  axis off;
subplot(1,3,2); imagesc(Im_InP);colormap gray; axis image;  axis off;
subplot(1,3,3); imagesc(Im_Rec); colormap gray; axis image; axis off;
suptitle(sprintf('PSNR:%0.2f, SSE:%0.2f',OutPSNR,OutSSE));










