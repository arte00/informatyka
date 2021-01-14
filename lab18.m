

A = [3 2 1; 5 4 3; 1 1 0; -1 0 0; 0 -1 0; 0 0 -1];
b = [120; 300; 50; 0; 0; 0];

F = [12 8 10];


matrix_rows = 1:6;
allComb = nchoosek(matrix_rows, 3);

sz = length(allComb);
results = zeros(3, sz);

for i=1:sz
    Ap = [A(allComb(i, 1),:); A(allComb(i, 2),:); A(allComb(i, 3),:)];
    bp = [b(allComb(i, 1)); b(allComb(i, 2)); b(allComb(i, 3))];
    results(:, i) = linsolve(Ap, bp);
end


final_results = [];
index_counter = 1;
eps = 1.0e-13;

for i=1:sz
    if ((A(1, :) * results(:, i) - b(1) <= eps) && ...
    (A(2, :) * results(:, i) - b(2) <= eps) && ...
    (A(3, :) * results(:, i) - b(3) <= eps) && ...
    (A(4, :) * results(:, i) - b(4) <= eps) && ...
    (A(5, :) * results(:, i) - b(5) <= eps) && ...
    (A(6, :) * results(:, i) - b(6) <= eps))
    final_results(:, index_counter) = results(:,i);
    index_counter = index_counter + 1;
    end
end


final_values = zeros(1, length(final_results));

for i=1:length(final_results)
    final_values(i) = F * final_results(:, i);
end

[res, res_index] = max(final_values);

x1 = final_results(1, res_index);
x2 = final_results(2, res_index);
x3 = final_results(3, res_index);

fprintf('Maksimum = %d\n', res)
fprintf('x1 =  %.2f\nx2 =  %.2f\nx3 =  %.2f\n', x1, x2, x3)


trisurf(convhull(A), final_results(1,:), final_results(2, :), final_results(3, :))
hold on
scatter3(x1, x2, x3, 'r')







