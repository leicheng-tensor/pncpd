function Z= proj(Z)
index=Z(:)<0;
Z(index)=0;
end