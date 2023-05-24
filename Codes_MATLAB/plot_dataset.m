function plot_dataset(X, X_label, Time, variables, units, titlename, xlabelname, ylim_lb, ylim_ub, if_saveplot, if_showplot, fname, if_nrmse, nrmse, numrow)

    assert(size(X,1) == numel(Time), ["Length of observations: ", num2str(numel(Time)),  " does not match with the number of observations: ", num2str(size(X,1))]);
    assert(size(X,2) == numel(variables), ["Length of variable: ", num2str(numel(variables)),  " does not match with the number of variables: ", num2str(size(X,2))]);
    
    if if_showplot == true
        
        figure(); clf();
        color = ["blue", "cyan", "red", "green", "magenta"];
        marker = ["o", "^", "*", "^", "*"];
        markersize = [20, 30, 70, 30, 70];
        numcol = ceil(size(X,2) / numrow);

        t = tiledlayout(numrow, numcol);
        set(gcf, 'WindowState', 'maximized')

        if if_saveplot == false
            sgtitle(titlename, 'FontSize', 30);
        end

        varind = 1;
        for i = 1:numrow
            for j = 1:numcol
                if varind > size(X,2)
                    break
                end
                nexttile
                for k = 1:5
                    x = Time(X_label(:, varind) == k-1);
                    y = X(X_label(:,varind) == k-1, varind);
                    scatter(x, y, markersize(k), marker(k), 'MarkerEdgeColor', color(k));
                    hold on
                end
                grid on
                grid minor
                if if_nrmse == 1
                    if nrmse(varind) > 0
                        title([variables{varind}, ' (', units{varind}, ')\nNRMSE: ', num2str(round(nrmse(varind), 4))], 'FontSize', 18);
                    else
                        title([variables{varind}, ' (', units{varind}, ')\nNRMSE: -'], 'FontSize', 18);
                    end
                else
                    title([variables{varind}, ' (', units{varind}, ')'], 'FontSize', 18);
                    
                end
                xlim([0, max(Time) + 1])
                ylim([ylim_lb(varind), ylim_ub(varind)])
                %set(gca, 'XTick', 1:V1, 'XTickLabel', xticklabels)
                %axs[i, j].tick_params(axis='x', labelsize=12)
                %axs[i, j].tick_params(axis='y', labelsize=12)
                varind = varind + 1;
            end
        end
        xlabel(t, xlabelname,  'FontSize', 24)
        box on
                    
        if if_saveplot == true
            plt.savefig([fname, titlename, '.png'])
        end
    end
end