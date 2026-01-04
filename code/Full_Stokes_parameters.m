
close all;clear all;

x = 250;
y = 230;
pixels = x*y;
filename = '28.mat';%%input the filename

%% non-denosed images
load(filename)
P = zeros(4,4096);
p1 = squeeze(Norm_photon(:,:,1));
p2 = squeeze(Norm_photon(:,:,2));
p3 = squeeze(Norm_photon(:,:,3));
p4 = squeeze(Norm_photon(:,:,4));
P = double([p1(:)';p2(:)';p3(:)';p4(:)']);
%%

%% after denoising 
% load(filename)
% p1 = squeeze(output(:,:,1));
% p1 = p1(:)';
% p2 = squeeze(output(:,:,2));
% p2 = p2(:)';
% p3 = squeeze(output(:,:,3));
% p3 = p3(:)';
% p4 = squeeze(output(:,:,4));
% p4 = p4(:)';
% P = double([p1 ; p2 ; p3 ; p4]);
%%

load('F12.mat');

f = pinv(F);
S = f*P;

s0 = reshape(S(1,:),y,x);
s1 = reshape(S(2,:),y,x);
s2 = reshape(S(3,:),y,x);
s3 = reshape(S(4,:),y,x);


DOP = (s1.^2+s2.^2+s3.^2).^(0.5)./s0;
DOLP = (s1.^2+s2.^2).^(1/2)./s0;
DOCP = abs(s3)./s0;

AOP = zeros();

CR = p1+p2+p3+p4;

for j = 1:1:x*y

    if S(3,j)>=0 && S(2,j)>=0
        AoP(j)=atand(S(3,j)/S(2,j));
    elseif S(3,j)>=0 && S(2,j)<0
        AoP(j)=(atand(S(3,j)/S(2,j))+180);
    elseif S(3,j)<0 && S(2,j)<0
        AoP(j)=(atand(S(3,j)/S(2,j))-180);
    elseif S(3,j)<0 && S(2,j)>0
        AoP(j)=(atand(S(3,j)/S(2,j)));
    end
end
AOP = reshape(AoP(:),y,x)/2;



AOP(isnan(AOP)) = 90;

DOCP(isnan(DOCP)) = 1;
DOCP(DOCP(:)>1)=1;
DOCP(DOCP(:)<0)=0;

DOLP(isnan(DOLP)) = 1;
DOLP(DOLP(:)>1)=1;
DOLP(DOLP(:)<0)=0;

DOP(isnan(DOP)) = 1;
DOP(DOP(:)>1)=1;
DOP(DOP(:)<0)=0;









