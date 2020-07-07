function out = normalizationimg(img)

ind = find(img~=0);
img2 = img(ind);

out1d = (img2-min(img2(:)))/(max(img2(:))-min(img2(:))+10e-8);
out = zeros(size(img));
out(ind)=out1d;
