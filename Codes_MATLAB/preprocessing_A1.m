function Xout = preprocessing_A1(Xin, time, process_method)
    Xout = Xin;
    switch process_method
        case 'mean'
            for j = 1:size(Xin,2)
                colmean = mean(Xin(:,j), 'omitnan');                
                Xout(isnan(Xin(:,j)), j) = colmean;             
            end
        case 'interpolation'
            for j = 1:size(Xin,2)
                CC = bwconncomp(isnan(Xin(:,j)));
                for k = 1:CC.NumObjects
                    indnan = CC.PixelIdxList{k};
                    if indnan(1) == 1
                        for l = 1:length(indnan)
                            Xout(length(indnan) - l + 1,j) = Xout(length(indnan) - l + 2,j);
                        end
                    elseif indnan(end) == length(time)
                        for l = 1:length(indnan)
                            Xout(indnan(l),j) = Xout(indnan(1)-1,j);
                        end
                    else
                        time_init = time(indnan(1)-1);
                        time_final = time(indnan(end)+1);
                        value_init = Xin(indnan(1)-1,j);
                        value_final = Xin(indnan(end)+1,j);
                        for l = 1:length(indnan)
                            Xout(indnan(l),j) = value_init + (value_final - value_init) / (time_final - time_init) * (time(indnan(l)) - time_init);
                        end
                    end
                end
            end
        case 'last_observed'
            for j = 1:size(Xin,2)
                CC = bwconncomp(isnan(Xin(:,j)));
                for k = 1:CC.NumObjects
                    indnan = CC.PixelIdxList{k};
                    for l = 1:length(indnan)
                        Xout(indnan(l),j) = Xin(indnan(1)-1,j);
                    end               
                end
            end
    end
end