
% info z zadania

a = [0.1 0.2 1 0 0; 0.3 0.1 0 1 0; 0.5 0 0 0 1];
b = [300; 300; 400];
Cb = zeros(3, 1);
Cj = [90 55 0 0 0];
bot_row = ones(1,6);

% wektory a1, a2, a3 ...
% i wektory bazowe

vec = [1 2 3 4 5];
base_vec = [3 4 5];

% wiersz wska�nik�w dla pierwszej tablicy

bot_row(1) = Cb'*b;

for i=1:5
    bot_row(i+1) = Cj(i) - Cb'*a(:,i);
end

while max(bot_row(2:6)) > 0
    
    % Znalezienie kolumny kluczowej
    
    [~, pivot_col] = max(bot_row(2:6));
    
    % Znalezienie wiersza kluczowego (B / Aj) i elementu rozwi�zuj�cego
    
    temp = zeros(1, 3);
    
    for i=1:3
        if a(i, pivot_col) > 0
            temp(i) = b(i) / a(i, pivot_col);
        else
            temp(i) = NaN;
        end
    end

    [~, pivot_row] = min(temp);
    pivot_element = a(pivot_row, pivot_col);
    
    % Zamiana wektor�w bazowych
    
    Cb(pivot_row) = Cj(pivot_col);
    base_vec(pivot_row) = vec(pivot_col);
    
    % dzielenie wiersza kluczowego przez element rozwi�zuj�cy
    
    a(pivot_row,:) = a(pivot_row,:) / pivot_element;
    b(pivot_row) = b(pivot_row) / pivot_element;

    % Zerowanie warto�ci w kolumnie kluczowej
    
    how_many_times = a(:, pivot_col) / a(pivot_row, pivot_col);
    how_many_times(pivot_row) = 0;

    for i=1:length(how_many_times)
        a(i, :) = a(i, :) - how_many_times(i)*a(pivot_row,:);
        b(i) = b(i) - how_many_times(i)*b(pivot_row);
    end
    
    % Aktualizacja wiersza wska�nik�w
    
    bot_row(1) = Cb'*b;
    
    for i=1:5
        bot_row(i+1) = Cj(i) - Cb'*a(:,i);
    end
    
end

% x1- szampon naturalny, x2 - szampon rodzinny

x1_index = find(base_vec==1);
x2_index = find(base_vec==2);

fprintf("\nFirma osi�gnie najwi�ksze zyski je�eli wyprodukuje %d litr�w szamponu ", b(x1_index))
fprintf("naturalnego i %d litr�w szamponu rodzinnego.\n", b(x2_index));
