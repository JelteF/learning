% Jelte Fennema 10183159
% Antwoord op deelvraag 7: Aan de grafiek van de gemiddelde fout is goed te
% zien dat de grootte van het huis de variabele is die het het best voorspelt

function main
    data = csvread('housesRegr.csv', 1, 0);
    mls = data(:, 1);
    bed = data(:, 2);
    bath = data(:, 3);
    size_ = data(:, 4);
    price = data(:, 5);
    do_the_stuff([5; 5; 3], [6; 6; 10], 'Writing assignment') % De data van de schrijf opdracht
    do_the_stuff(mls, price, 'MLS')
    do_the_stuff(bed, price, 'Bedrooms')
    do_the_stuff(bath, price, 'Bathrooms')
    do_the_stuff(size_, price, 'Size in square feet')
end

function do_the_stuff(x_vec, y_vec, plot_title)
    hold off
    subplot(1,2,1)
    hold off
    scatter(x_vec, y_vec)
    title(plot_title)
    hold on
    theta0 = 0;
    theta1 = 0;

    alpha = 1;
    terminate = 500;

    % Gebruik normalisatie tussen -1 en 1 om dezelfde alpha goed te laten zijn
    % voor de verschillende variabelen.
    x_vec_old = x_vec;
    x_vec = normalise(x_vec);

    costs = [J_vec(x_vec, y_vec, theta0, theta1)];

    for i=1:terminate
        [grad_theta0, grad_theta1] = gradient_vec(x_vec, y_vec, theta0, theta1);
        theta0 -= alpha * grad_theta0;
        theta1 -= alpha * grad_theta1;
        costs = [costs J_vec(x_vec, y_vec, theta0, theta1)];
    end
    plot(x_vec_old, theta0 + theta1 * x_vec)
    subplot(1,2,2)
    % Extra plot that shows
    plot(0:terminate, costs);
    title('Mean squared error after x iterations');
    ylabel('Error');
    xlabel('Iterations');
    pause(5);
end

function ret = normalise(vec)
    ret = (vec - mean(vec))/(max(vec) - min(vec));
end

function [grad_theta0, grad_theta1] = gradient_iter(x_vec, y_vec, theta0, theta1)
    m = length(x_vec);
    errors = 0;
    grad_theta0 = 0;
    grad_theta1 = 0;
    for i=1:length(x_vec)
        prediction = theta0 + theta1 * x_vec(i);
        err = prediction-y_vec(i);
        grad_theta0 += err;
        grad_theta1 += err * x_vec(i);
    end
    grad_theta0 = (1/m) * grad_theta0;
    grad_theta1 = (1/m) * grad_theta1;
end

function [grad_theta0, grad_theta1] = gradient_vec(x_vec, y_vec, theta0, theta1)
    m = length(x_vec);
    err = theta0 + theta1 * x_vec - y_vec;
    grad_theta0 = (1/m) * sum(err);
    grad_theta1 = (1/m) * sum(err .* x_vec);
end

function ret = J_iter(x_vec, y_vec, theta0, theta1)
    m = length(x_vec);
    errors = 0;
    for i=1:length(x_vec)
        prediction = theta0 + theta1 * x_vec(i);
        errors += (prediction-y_vec(i))^2;
    end
    ret = 1/(2*m) * errors;
end

function ret = J_vec(x_vec, y_vec, theta0, theta1)
    m = length(x_vec);
    predictions = theta0 + x_vec * theta1;
    errors = (predictions-y_vec).^2;
    ret = 1/(2*m) * sum(errors);
end

main
