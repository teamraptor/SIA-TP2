function cmap = get_map(mymap)
    n = 256; % size of new color map
    m = size(mymap,1);
    t0 = linspace(0,1,m)';
    t = linspace(0,1,n)';
    r = interp1(t0,mymap(:,1),t);
    g = interp1(t0,mymap(:,2),t);
    b = interp1(t0,mymap(:,3),t);
    cmap = [r,g,b];
end