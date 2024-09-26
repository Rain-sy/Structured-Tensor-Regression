function penalty_single = scad_penalty(x, l, a)
    if abs(x) <= l
        penalty_single = l * abs(x);
    elseif abs(x) <= a * l
        penalty_single = (-x^2 + 2 * a * l * abs(x) - l^2) / (2 * (a - 1));
    else
        penalty_single = (l^2 * (a^2 - 1)) / 2;
    end
end