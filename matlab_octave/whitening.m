function [Z, W] = whitening(Y, mode, n_components)
    % Whitens the data Y using sphering or pca
    R = (Y * Y') / size(Y, 2);
    [U, D, ~] = svd(R);
    D = diag(D);
    if strcmp(mode, 'pca')
        W = diag(1. ./ sqrt(D)) * U';
        W = W(1:n_components, :);
        for i = 1:size(W,1)
            [~, j] = max(abs(W(i,:)));
            if W(i,j) < 0
                W(i,:) = -W(i,:);
            end
        end
        Z = W * Y;
    elseif strcmp(mode, 'sph')
        W = U *  diag(1. ./ sqrt(D)) * U';
        Z = W * Y;
    end
end
