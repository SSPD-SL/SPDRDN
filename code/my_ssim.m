
    function s = my_ssim(img1, img2, L)
    K = [0.01 0.03];
    %L = 180;
    C1 = (K(1)*L)^2;
    C2 = (K(2)*L)^2;

    mu1 = mean2(img1);
    mu2 = mean2(img2);
    sigma1_sq = var(img1(:));
    sigma2_sq = var(img2(:));
    sigma12 = cov(double(img1(:)), double(img2(:)));
    sigma12 = sigma12(1,2);

    s = (2*mu1*mu2 + C1)*(2*sigma12 + C2) / ...
        ((mu1^2 + mu2^2 + C1)*(sigma1_sq + sigma2_sq + C2));
    end