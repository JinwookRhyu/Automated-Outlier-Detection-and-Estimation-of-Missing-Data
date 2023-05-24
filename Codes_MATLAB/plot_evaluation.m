function plot_evaluation(feasibility, plausibility, time_final, fname, titlename, if_saveplot, if_showplot)

    if if_showplot == true
        
        figure(); clf();
        t = tiledlayout(1, 3);
        set(gcf, 'WindowState', 'maximized')

        
        sgtitle('Algorithm evaluation results', fontsize=40)
        algorithms = {'MI', 'ALS', 'Alternating', 'SVDImpute', 'PCADA', 'PPCA', 'PPCA-M', 'BPCA', 'SVT', 'ALM'};
        xpos = 1:length(algorithms);
        cmap = [0.5 0.5 0.5; 1 0 0; 1 0.33 0.1; 1 0.66 0.2; 1 1 0; 0.5 1 0; 0 0.5 0; 0 1 1; 0 0 1; 0.5 0 0.5];
      
        nexttile
        b = bar(xpos, feasibility);
        b.FaceColor = 'flat';
        for i = 1:10
            b.CData(i,:) = cmap(i,:);
        end
        grid on
        grid minor
        ylim([min(0, min(feasibility) * 0.8), max(1, max(feasibility) * 1.2)])
        set(gca,'Xticklabel',[])
        title('Feasibility', 'FontSize', 25)
        ylabel('# of imputed values outside the boundaries', 'FontSize', 15)

        nexttile
        b = bar(xpos, plausibility);
        b.FaceColor = 'flat';
        for i = 1:10
            b.CData(i,:) = cmap(i,:);
        end
        grid on
        grid minor
        ylim([min(0, min(plausibility) * 0.8), max(1, max(plausibility) * 1.2)])
        set(gca,'Xticklabel',[])
        title('Plausibility', 'FontSize', 25)
        ylabel('# of imputed values considered outliers', 'FontSize', 15)

        nexttile
        b = bar(xpos, time_final);
        b.FaceColor = 'flat';
        for i = 1:10
            b.CData(i,:) = cmap(i,:);
        end
        ylim([min(10^(-3), min(time_final) * 0.8), max(1, max(time_final) * 1.2)])
        grid on
        grid minor
        set(gca,'YScale','log')
        set(gca,'Xticklabel',[])
        title('Rapidity', 'FontSize', 25)
        ylabel('Computation time (s)', 'FontSize', 15)

                    
        if if_saveplot == true
            plt.savefig([fname, titlename, '.png'])
        end
    end
end